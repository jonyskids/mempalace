# Loaded into mempalace.mcp_server via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.mcp_server":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.mcp_server")


# ==================== KNOWLEDGE GRAPH ====================

# Max facts returned by a single kg_query response. A broad query (e.g. a hub
# entity with thousands of relationships) can otherwise return hundreds of
# thousands of characters and overflow the caller's context window. Override
# via MEMPALACE_KG_RESULT_CAP. (kg_timeline is bounded by its own upstream
# limit/offset pagination, #2406.)
try:
    KG_RESULT_CAP = max(1, int(os.environ.get("MEMPALACE_KG_RESULT_CAP", "100")))
except (ValueError, TypeError):
    KG_RESULT_CAP = 100

# Hard ceiling (in characters) on any single serialized tool response, applied
# at the dispatch chokepoint as a backstop against context overflow. ~50KB.
# Override via MEMPALACE_MAX_RESPONSE_CHARS.
try:
    MAX_RESPONSE_CHARS = max(1000, int(os.environ.get("MEMPALACE_MAX_RESPONSE_CHARS", "50000")))
except (ValueError, TypeError):
    MAX_RESPONSE_CHARS = 50000


def _cap_facts(results: list, *, narrow_hint: str) -> tuple[list, Optional[str]]:
    """Cap a fact list to ``KG_RESULT_CAP``; return (kept, truncation_notice).

    Returns the full list unchanged (and ``None`` notice) when within the cap.
    Over the cap, returns the first ``KG_RESULT_CAP`` facts plus a notice the
    caller surfaces in the response so truncation is never silent.
    """
    if not isinstance(results, list) or len(results) <= KG_RESULT_CAP:
        return results, None
    omitted = len(results) - KG_RESULT_CAP
    notice = (
        f"{omitted} more fact(s) omitted (showing first {KG_RESULT_CAP} of "
        f"{len(results)}); {narrow_hint}"
    )
    return results[:KG_RESULT_CAP], notice


def _temporal_bound_key(value, *, end: bool = False) -> Optional[str]:
    if not value:
        return None
    text = str(value)
    if "T" in text:
        return text
    return f"{text}T23:59:59Z" if end else f"{text}T00:00:00Z"


def _fact_interval_bucket(row: dict, now_key: str) -> str:
    start_key = _temporal_bound_key(row.get("valid_from"), end=False)
    end_key = _temporal_bound_key(row.get("valid_to"), end=True)
    if start_key and start_key > now_key:
        return "future"
    if end_key and end_key < now_key:
        return "historical"
    return "active"


# Max distinct predicates listed in the histogram attached to a dead-end
# kg_query response. A hub entity can carry well over a hundred distinct
# predicates (project:digital-p had 177 on 2026-08-23), most of them
# singletons; the top slice by fact count is what a caller actually re-queries
# with, and the tail would crowd out the facts themselves.
KG_PREDICATE_HISTOGRAM_LIMIT = 25


# Max predicates listed in a SUMMARY histogram. This one is deliberately far
# larger than KG_PREDICATE_HISTOGRAM_LIMIT: the summary histogram's entire job
# is that nothing on the entity is invisible, so it lists every predicate and
# only degrades on an entity pathological enough to threaten the response cap
# on histogram rows alone (project:digital-p carried 187 on 2026-08-24). When
# it does degrade it says so, in the same explicit style as every other
# truncation here. Override via MEMPALACE_KG_SUMMARY_HISTOGRAM_LIMIT.
try:
    KG_SUMMARY_HISTOGRAM_LIMIT = max(
        1, int(os.environ.get("MEMPALACE_KG_SUMMARY_HISTOGRAM_LIMIT", "500"))
    )
except (ValueError, TypeError):
    KG_SUMMARY_HISTOGRAM_LIMIT = 500


def _normalize_predicate_filter(predicate):
    """Normalize the ``predicate=`` argument of ``tool_kg_query``.

    Accepts a single name ("completed"), a comma-separated string
    ("completed,shipped") or a list of names. Returns
    ``(predicates_or_None, error_or_None)``; ``(None, None)`` means the
    caller passed no filter and the query must behave exactly as it did
    before this parameter existed.
    """
    if predicate is None:
        return None, None

    if isinstance(predicate, str):
        raw_items = predicate.split(",")
    elif isinstance(predicate, (list, tuple)):
        raw_items = []
        for item in predicate:
            if not isinstance(item, str):
                return None, "predicate list entries must be strings"
            raw_items.extend(item.split(","))
    else:
        return None, "predicate must be a string or a list of strings"

    normalized = []
    for item in raw_items:
        item = item.strip()
        if not item:
            continue
        try:
            item = sanitize_name(item, "predicate")
        except ValueError as e:
            return None, str(e)
        item = item.lower().replace(" ", "_")
        if item not in normalized:
            normalized.append(item)

    if not normalized:
        return None, "predicate must name at least one predicate"
    return normalized, None


def _predicate_histogram(results, limit: Optional[int] = None):
    """Return ``(top_counts, omitted_predicates, distinct_predicates)``.

    Computed from facts already in hand, so it always describes exactly the
    ``direction`` / ``as_of`` slice the caller asked about. ``limit`` is
    resolved from module state at call time (not bound as a default) so an
    override takes effect for the call that sets it.
    """
    if limit is None:
        limit = KG_PREDICATE_HISTOGRAM_LIMIT
    counts: dict = {}
    for fact in results:
        if not isinstance(fact, dict):
            continue
        pred = fact.get("predicate")
        if pred is None:
            continue
        counts[pred] = counts.get(pred, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = dict(ordered[:limit])
    return top, len(ordered) - len(top), len(ordered)


def _summarize_facts(results, *, budget: int, histogram_limit: Optional[int] = None):
    """Derive a read-time summary of ``results``. Returns ``(sample, summary)``.

    Everything here is computed from the facts already in hand, so it describes
    exactly the ``direction`` / ``as_of`` / ``predicate`` slice the caller asked
    about and can never be stale. There is no summary table and no cache — that
    is the point.

    Two failures of a plain result cap are addressed:

    * **Blended validity.** ``query_entity`` applies no ``valid_to`` filter
      unless ``as_of`` is given, so a raw count of a hub predicate counts dead
      facts alongside live ones and over-reports what is actually open
      (``project:digital-p`` / ``pending_task``: 335 rows, 93 of them current on
      2026-08-24). Every count here is split ``[current, invalidated]``.
    * **A silent cut mid-set.** The histogram names EVERY predicate on the
      entity with its counts, so a predicate can be unsampled but never
      invisible, and the caller always has the exact name to re-query with.

    The sample is allocated round-robin: predicates in histogram order (widest
    live slice first) each yield their most recent current fact, then their
    second, and so on until ``budget`` is spent. Current facts are exhausted
    before invalidated ones within a predicate, so a predicate with live facts
    never spends its slot on a dead one. There is deliberately no fixed
    per-predicate N: round-robin derives it from the shape of the entity.
    A single-predicate query spends the whole budget on depth (the 335-row
    ``pending_task`` slice comes back as its 93 current facts plus the most
    recent dead ones); a 187-predicate query spends it on breadth.
    """
    if histogram_limit is None:
        histogram_limit = KG_SUMMARY_HISTOGRAM_LIMIT

    buckets: dict = {}
    for fact in results:
        if not isinstance(fact, dict):
            continue
        pred = fact.get("predicate")
        if pred is None:
            continue
        bucket = buckets.setdefault(pred, {"current": [], "invalidated": []})
        bucket["current" if fact.get("valid_to") is None else "invalidated"].append(fact)

    # Live slices first: a caller asking a hub entity "what is open" is served
    # worse by a predicate with 200 dead facts than by one with 3 live ones.
    ordered = sorted(
        buckets.items(),
        key=lambda kv: (
            -len(kv[1]["current"]),
            -(len(kv[1]["current"]) + len(kv[1]["invalidated"])),
            kv[0],
        ),
    )

    queues = {pred: bucket["current"] + bucket["invalidated"] for pred, bucket in ordered}
    picked: dict = {pred: [] for pred, _ in ordered}
    taken = 0
    depth = 0
    while taken < budget:
        progressed = False
        for pred, _ in ordered:
            if taken >= budget:
                break
            queue = queues[pred]
            if depth < len(queue):
                picked[pred].append(queue[depth])
                taken += 1
                progressed = True
        if not progressed:
            break
        depth += 1

    # Emit grouped by predicate rather than in round-robin pick order, so a
    # reader sees each slice together instead of interleaved.
    sample = []
    for pred, _ in ordered:
        sample.extend(picked[pred])

    histogram = {
        pred: [len(bucket["current"]), len(bucket["invalidated"])]
        for pred, bucket in ordered[:histogram_limit]
    }
    sampled = sum(1 for pred in picked if picked[pred])

    summary = {
        "total": len(results),
        "current": sum(len(b["current"]) for b in buckets.values()),
        "invalidated": sum(len(b["invalidated"]) for b in buckets.values()),
        "predicates_distinct": len(ordered),
        "histogram_format": "predicate: [current_facts, invalidated_facts]",
        "histogram": histogram,
        "facts_shown": len(sample),
        "predicates_sampled": sampled,
        "selection_rule": (
            f"'facts' is a derived sample, not a prefix: up to {budget} facts allocated "
            "round-robin across predicates (widest live slice first), current facts before "
            "invalidated ones, most recent first within each group. 'histogram' counts ALL "
            "facts on this entity by predicate, so nothing is hidden. Re-query with "
            "predicate='<name>' to get one slice whole."
        ),
    }

    unsampled = len(ordered) - sampled
    if unsampled:
        summary["predicates_unsampled"] = unsampled
    if len(ordered) > histogram_limit:
        summary["histogram_omitted"] = len(ordered) - histogram_limit
        summary["histogram_notice"] = (
            f"{len(ordered) - histogram_limit} rarer predicate(s) not listed in 'histogram' "
            f"(showing the {histogram_limit} with the most current facts, of "
            f"{len(ordered)}); raise MEMPALACE_KG_SUMMARY_HISTOGRAM_LIMIT to see them all"
        )
    return sample, summary


def _interval_buckets(facts: list, as_of) -> tuple[list, list, list]:
    """Split shipped facts into v3.10's active / historical / future buckets."""
    if as_of is not None:
        return facts, [], []
    now_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    active = []
    historical = []
    future = []
    for row in facts:
        bucket = _fact_interval_bucket(row, now_key)
        if bucket == "future":
            future.append(row)
        elif bucket == "historical":
            historical.append(row)
        else:
            active.append(row)
    return active, historical, future


def _kg_query_base_payload(entity: str, as_of, facts: list) -> dict:
    active, historical, future = _interval_buckets(facts, as_of)
    return {
        "entity": entity,
        "as_of": as_of,
        "active_facts": active,
        "historical_facts": historical,
        "future_facts": future,
        "facts": facts,
        "count": len(facts),
    }


def _kg_query_resolve_entity(payload: dict, entity: str, results: list, *, lookup_candidates: bool):
    """Upstream (v3.10) entity resolution: resolved_from, or candidates on a miss."""
    if results:
        resolved_names = {
            r.get("subject") if r.get("direction") == "outgoing" else r.get("object")
            for r in results
        }
        resolved_names.discard(None)
        if len(resolved_names) == 1:
            resolved = next(iter(resolved_names))
            if resolved != entity:
                payload["resolved_from"] = entity
                payload["entity"] = resolved
    elif lookup_candidates:
        candidates = _call_kg(lambda kg: kg.find_entity_candidates(entity))
        if candidates:
            payload["candidates"] = candidates


def _fits_response_cap(response) -> bool:
    return len(json.dumps(response, indent=2, ensure_ascii=False)) <= MAX_RESPONSE_CHARS


def _largest_fitting(build, hi: int):
    """Return ``build(n)`` for the largest ``n <= hi`` whose response fits.

    Serialized size grows monotonically with the fact budget, so bisection
    finds it. When even ``build(1)`` is over the cap, ``build(1)`` is returned
    and the dispatch backstop (``_fit_response``) has the last word.
    """
    response = build(hi)
    if hi <= 1 or _fits_response_cap(response):
        return response
    best = None
    lo, hi = 1, hi - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = build(mid)
        if _fits_response_cap(trial):
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best is not None else build(1)


def tool_kg_query(
    entity: str,
    as_of: str = None,
    direction: str = "both",
    predicate=None,
    summary=None,
):
    """Query the knowledge graph for an entity's relationships.

    ``predicate`` filters to one or more relationship types. It composes with
    ``direction`` and ``as_of`` in a single SQL query rather than forking a
    second path, and is a pure no-op when omitted.

    A hub entity can carry thousands of facts, far past both the
    ``KG_RESULT_CAP`` fact cap and the serialized response cap. Truncation is
    always announced. When an *unfiltered* query is truncated, or when a
    *filtered* query matches nothing, the response also carries a predicate
    histogram and an instruction, so a dead end becomes a next query instead
    of a silent absence the caller misreads as "no such fact".

    ``summary`` controls summary mode, which replaces an arbitrary truncated
    prefix with a complete per-predicate histogram (split current vs
    invalidated) plus a representative sample of the facts themselves. See
    ``_summarize_facts``. Three states:

    * ``None`` (default) — automatic, and only when the result overflows
      ``KG_RESULT_CAP``. A response that fits is untouched: same keys, same
      facts, same order as before this parameter existed.
    * ``True`` — always summarize, even under the cap. This is how a caller
      asks "what does this entity carry?" without pulling every fact.
    * ``False`` — never summarize. Escape hatch back to the plain recent-first
      prefix for a caller that wants exactly the old truncation behaviour.

    Automatic-on-overflow rather than opt-in-only because the failure it
    fixes is one the caller cannot detect: an over-cap response looked
    authoritative, and a count that blended dead facts with live ones
    over-reported what was open. A caller who has to know to ask is a caller
    who is still misled by default. Opt-in as well because a summary of an
    entity that fits is genuinely useful and nothing else offers it.

    v3.10 ships every fact twice — in ``facts`` and again in one of the
    ``active_facts`` / ``historical_facts`` / ``future_facts`` buckets — and
    the buckets are always derived from the facts that ship. Both the plain
    prefix and the summary sample are therefore sized to
    ``MAX_RESPONSE_CHARS`` here, by bisecting the fact budget, rather than
    left to ``_fit_response``: the generic backstop drops items off the tail
    of the single largest list, which would cut ``facts`` while leaving its
    bucket copies whole (and, for a grouped summary sample, silently delete
    the rarest predicates' only fact and relabel it "showing first N").
    """
    try:
        entity = sanitize_kg_value(entity, "entity")
        as_of = sanitize_iso_temporal(as_of, "as_of")
    except ValueError as e:
        return {"error": str(e)}

    if direction not in ("outgoing", "incoming", "both"):
        return {"error": "direction must be 'outgoing', 'incoming', or 'both'"}

    if summary is not None and not isinstance(summary, bool):
        return {"error": "summary must be true or false"}

    predicates, predicate_error = _normalize_predicate_filter(predicate)
    if predicate_error is not None:
        return {"error": predicate_error}

    results = _call_kg(
        lambda kg: kg.query_entity(entity, as_of=as_of, direction=direction, predicates=predicates)
    )
    if not isinstance(results, list):
        results = []
    total = len(results)

    if predicates:
        narrow_hint = (
            f"this is ALREADY a predicate-filtered slice ({', '.join(predicates)}) and it "
            f"is STILL over the {KG_RESULT_CAP}-fact cap — narrow further with a single "
            "predicate=, or with as_of= / direction="
        )
    else:
        narrow_hint = (
            "narrow with predicate= (the 'predicates' field in this response lists the "
            "available slices and their fact counts), as_of= or direction="
        )
    over_cap = total > KG_RESULT_CAP
    summarizing = bool(results) and (summary is True or (summary is None and over_cap))

    # A filtered miss is answered from the unfiltered slice; fetch it once,
    # not once per bisection step.
    unfiltered = None
    if predicates and not results:
        unfiltered = _call_kg(lambda kg: kg.query_entity(entity, as_of=as_of, direction=direction))

    def assemble(facts, notice, summary_block):
        response = _kg_query_base_payload(entity, as_of, facts)
        if predicates:
            response["predicate"] = predicates
        if notice is not None:
            response["total"] = total
            response["truncated"] = True
            response["notice"] = notice
        if summary_block is not None:
            response["summary"] = summary_block

        # Rescue the two dead ends: an unfiltered query too big to return whole,
        # and a filtered query that matched nothing (usually a predicate synonym).
        histogram_source = None
        hint = None
        if not predicates and notice is not None:
            histogram_source = results
            if summary_block is not None:
                hint = (
                    f"Showing a representative {len(facts)} of {total} facts, sampled across "
                    "predicates. Re-query with predicate='<name>' (or a comma-separated list) "
                    "to get one COMPLETE slice instead. 'summary.histogram' counts every "
                    "predicate on this entity split current/invalidated; 'predicates' is the "
                    "top slice of it by raw fact count."
                )
            else:
                hint = (
                    f"Showing an arbitrary first {len(facts)} of {total} facts. Re-query "
                    "with predicate='<name>' (or a comma-separated list) to get one COMPLETE "
                    "slice instead. 'predicates' lists this entity's predicates by fact count."
                )
        elif isinstance(unfiltered, list) and unfiltered:
            histogram_source = unfiltered
            hint = (
                f"No fact on this entity uses predicate {', '.join(predicates)} in this "
                "direction/as_of slice. 'predicates' lists the predicates it DOES use — "
                "check for a synonym there before writing a new fact."
            )

        # Entity candidates only when the entity itself carries nothing: a
        # filtered miss on an entity that DOES have facts is answered by the
        # predicate histogram, not by a list of similarly named entities.
        _kg_query_resolve_entity(response, entity, results, lookup_candidates=not histogram_source)

        if histogram_source:
            top, omitted, distinct = _predicate_histogram(histogram_source)
            response["predicates"] = top
            response["predicates_distinct"] = distinct
            if omitted:
                response["predicates_omitted"] = omitted
                hint = (
                    hint + f" ({omitted} rarer predicate(s) not listed; "
                    f"showing the top {len(top)} of {distinct}.)"
                )
            response["predicate_hint"] = hint

        return response

    # Cut for SIZE rather than for the fact cap: the facts fit the cap but not
    # the response, so "you are over the fact cap" would be false.
    size_hint = (
        f"the facts are too large to return whole under the {MAX_RESPONSE_CHARS}-char "
        "response cap — narrow with predicate=, as_of= or direction="
    )

    def plain(limit):
        if limit >= KG_RESULT_CAP:
            facts, notice = _cap_facts(results, narrow_hint=narrow_hint)
            return assemble(facts, notice, None)
        notice = (
            f"{total - limit} more fact(s) omitted (showing first {limit} of "
            f"{total}); {narrow_hint if over_cap else size_hint}"
        )
        return assemble(results[:limit], notice, None)

    def summarized(budget):
        facts, summary_block = _summarize_facts(results, budget=budget)
        notice = None
        if len(facts) < total:
            notice = (
                f"{total - len(facts)} more fact(s) not shown (showing {len(facts)} of "
                f"{total}: {summary_block['current']} current / "
                f"{summary_block['invalidated']} invalidated across "
                f"{summary_block['predicates_distinct']} predicate(s)). 'facts' is a "
                "representative sample and 'summary.histogram' counts every predicate on "
                f"this entity, so nothing is hidden — {narrow_hint if over_cap else size_hint}"
            )
        return assemble(facts, notice, summary_block)

    return _largest_fitting(summarized if summarizing else plain, KG_RESULT_CAP)


def tool_kg_add(
    subject: str,
    predicate: str,
    object: str,
    valid_from: str = None,
    valid_to: str = None,
    source_closet: str = None,
    source_file: str = None,
    source_drawer_id: str = None,
):
    """Add a relationship to the knowledge graph.

    All temporal and provenance fields are optional. ``valid_to`` lets callers
    backfill historical facts with a known end date/time in a single call
    instead of a separate ``kg_invalidate`` call.

    Temporal values accept either ``YYYY-MM-DD`` or canonical UTC datetimes in
    the form ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    try:
        subject = sanitize_kg_value(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_kg_value(object, "object")
        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "source_closet": source_closet,
            "source_file": source_file,
            "source_drawer_id": source_drawer_id,
        },
    )

    triple_id = _call_kg(
        lambda kg: kg.add_triple(
            subject,
            predicate,
            object,
            valid_from=valid_from,
            valid_to=valid_to,
            source_closet=source_closet,
            source_file=source_file,
            source_drawer_id=source_drawer_id,
        )
    )
    return {"success": True, "triple_id": triple_id, "fact": f"{subject} → {predicate} → {object}"}


def tool_kg_invalidate(subject: str, predicate: str, object: str, ended: str = None):
    """Mark a fact as no longer true.

    Returns ``matched`` — how many open facts were actually closed — and the
    actual ``ended`` date/time that was stored. When the caller omits
    ``ended``, the underlying graph stamps ``date.today()`` and the response
    reflects that resolved value.

    ``object`` is matched verbatim against stored facts. When it matches no
    open fact the response is ``success: False`` with ``matched: 0`` and
    ``error_class: "NoMatchingFact"``: nothing was written, and a caller that
    reads only ``success`` must not conclude the fact was retired.

    Temporal values accept either ``YYYY-MM-DD`` or canonical UTC datetimes in
    the form ``YYYY-MM-DDTHH:MM:SSZ``.

    Note: ``object`` is a lookup key against existing facts and is not subject
    to the 128-char entity-name limit — long free-text object strings are
    supported for invalidation.
    """
    try:
        subject = sanitize_kg_value(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        # object is a lookup key, not a new entity name — skip the 128-char
        # MAX_NAME_LENGTH cap so long free-text facts can be invalidated.
        if not isinstance(object, str) or not object.strip():
            raise ValueError("object must be a non-empty string")
        object = strip_lone_surrogates(object.strip())
        if "\x00" in object:
            raise ValueError("object contains null bytes")
        ended = sanitize_iso_temporal(ended, "ended")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    resolved_ended = ended or date.today().isoformat()

    _wal_log(
        "kg_invalidate",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "ended": resolved_ended,
        },
    )

    matched = _call_kg(lambda kg: kg.invalidate(subject, predicate, object, ended=resolved_ended))
    fact = f"{subject} → {predicate} → {object}"
    if not matched:
        # Nothing was retired. The old response returned success with this
        # same ``fact`` string -- which is the CALLER'S input echoed back,
        # never a stored row -- so a mistyped or re-worded object read as
        # confirmation while the stale fact stayed live.
        return _refused_write(
            "mempalace_kg_invalidate",
            (
                f"no open fact matched {fact} — nothing was invalidated. "
                "The object is matched verbatim against stored facts: it may "
                "already be ended, or the stored wording may differ. Check "
                "mempalace_kg_query before retrying."
            ),
            "NoMatchingFact",
            fact=fact,
            ended=resolved_ended,
            matched=0,
        )
    return {
        "success": True,
        "fact": fact,
        "ended": resolved_ended,
        "matched": matched,
    }


def tool_kg_supersede(
    subject: str,
    predicate: str,
    old_object: str,
    new_object: str,
    at: str = None,
):
    """Atomically replace one fact with another at a single shared boundary.

    Closes ``(subject, predicate, old_object)`` and opens
    ``(subject, predicate, new_object)`` at one shared instant, so a
    point-in-time query at the boundary returns only the new value. Use this
    instead of a separate ``kg_invalidate`` + ``kg_add`` when a single-valued
    fact changes (e.g. a model, employer, or address changes).

    ``at`` accepts ``YYYY-MM-DD`` or a canonical UTC datetime
    (``YYYY-MM-DDTHH:MM:SSZ``) and defaults to the current UTC instant.
    """
    try:
        subject = sanitize_kg_value(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        old_object = sanitize_kg_value(old_object, "old_object")
        new_object = sanitize_kg_value(new_object, "new_object")
        at = sanitize_iso_temporal(at, "at")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    _wal_log(
        "kg_supersede",
        {
            "subject": subject,
            "predicate": predicate,
            "old_object": old_object,
            "new_object": new_object,
            "at": at,
        },
    )

    # Domain ValueErrors from kg.supersede (e.g. inverted boundary) are left to
    # bubble to the dispatcher, matching tool_kg_add / tool_kg_invalidate: the
    # -32000 response carries error_class + message in error.data. Only input
    # sanitization above returns the {success: False} envelope.
    triple_id = _call_kg(lambda kg: kg.supersede(subject, predicate, old_object, new_object, at=at))
    return {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{subject} → {predicate} → {new_object}",
        "superseded": old_object,
    }


def tool_kg_timeline(entity: str = None, limit: int = 100, offset: int = 0):
    """Get chronological timeline of facts, optionally for one entity.

    Paginated with ``limit``/``offset`` following the ``tool_list_drawers``
    convention; defaults match the historical behavior (first 100 facts).
    """
    limit = max(1, min(limit, _MAX_RESULTS))
    offset = max(0, offset)
    if entity is not None:
        try:
            entity = sanitize_kg_value(entity, "entity")
        except ValueError as e:
            return {"error": str(e)}

    def _query(kg):
        return {
            "timeline": kg.timeline(entity, limit=limit, offset=offset),
            "total": kg.timeline_total(entity),
        }

    result = _call_kg(_query)
    return {
        "entity": entity or "all",
        "timeline": result["timeline"],
        "count": len(result["timeline"]),
        "total": result["total"],
        "offset": offset,
        "limit": limit,
    }


def tool_kg_stats():
    """Knowledge graph overview: entities, triples, relationship types."""
    return _call_kg(lambda kg: kg.stats())
