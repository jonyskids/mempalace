"""MCP server tests — knowledge-graph tools and tenant cache."""

import logging
import os
import subprocess
import sys


from _mcp_server_helpers import (
    _patch_mcp_server,
)


class TestKGTools:
    def test_kg_add(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="Alice",
            predicate="likes",
            object="coffee",
            valid_from="2025-01-01",
        )
        assert result["success"] is True

    def test_kg_query(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max")
        assert result["count"] > 0

    def test_kg_invalidate(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        result = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="chess",
            ended="2026-03-01",
        )
        assert result["success"] is True
        # Regression #1314: response must echo the actual ended date,
        # not silently drop it and return the literal string "today".
        assert result["ended"] == "2026-03-01"

        # The response must say how many open facts were actually closed,
        # so "success" is a measured outcome rather than a shape.
        assert result["matched"] == 1

    def test_kg_supersede(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_supersede

        kg.add_triple("Bot", "uses_model", "old", valid_from="2026-05-01")
        result = tool_kg_supersede(
            subject="Bot",
            predicate="uses_model",
            old_object="old",
            new_object="new",
            at="2026-06-02",
        )
        assert result["success"] is True
        assert result["superseded"] == "old"
        models = [
            f["object"]
            for f in kg.query_entity("Bot", as_of="2026-06-02", direction="outgoing")
            if f["predicate"] == "uses_model"
        ]
        assert models == ["new"]

    def test_kg_add_forwards_valid_to(self, monkeypatch, config, palace_path, kg):
        """Regression #1314 case 1: valid_to must round-trip through kg_add."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="_test_temporal",
            predicate="had_value",
            object="probe",
            valid_from="2026-01-01",
            valid_to="2026-04-28",
        )
        assert result["success"] is True

        facts = kg.query_entity("_test_temporal")
        assert len(facts) == 1
        assert facts[0]["valid_from"] == "2026-01-01"
        assert facts[0]["valid_to"] == "2026-04-28"
        # An already-ended fact must not be reported as still current.
        assert facts[0]["current"] is False

    def test_kg_add_forwards_source_provenance(self, monkeypatch, config, palace_path, kg):
        """Regression #1314 case 3: source_file / source_drawer_id reach storage."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="operating-verb",
            predicate="candidate",
            object="husbandry",
            valid_from="2026-04-28",
            source_closet="closet-42",
            source_file="docs/decisions.md",
            source_drawer_id="drawer_abc123",
        )
        assert result["success"] is True

        triple_id = result["triple_id"]
        # Read raw row to verify all provenance columns persisted.
        with kg._lock:
            row = (
                kg._conn()
                .execute(
                    "SELECT source_closet, source_file, source_drawer_id FROM triples WHERE id = ?",
                    (triple_id,),
                )
                .fetchone()
            )
        assert row is not None
        assert row["source_closet"] == "closet-42"
        assert row["source_file"] == "docs/decisions.md"
        assert row["source_drawer_id"] == "drawer_abc123"

    def test_kg_invalidate_returns_actual_ended_date(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """Regression #1314 case 2: response reports the resolved date, not 'today'."""
        from datetime import date as _date

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        # Caller-supplied date round-trips into the response.
        explicit = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="swimming",
            ended="2026-04-28",
        )
        assert explicit["ended"] == "2026-04-28"

        # Caller-omitted date resolves to today's ISO date — never the
        # literal string "today" the buggy implementation used to return.
        implicit = tool_kg_invalidate(
            subject="Max",
            predicate="loves",
            object="Chess",
        )
        assert implicit["ended"] != "today"
        assert implicit["ended"] == _date.today().isoformat()

    def test_kg_timeline(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_timeline

        result = tool_kg_timeline(entity="Alice")
        assert result["count"] > 0
        assert result["total"] >= result["count"]
        assert result["offset"] == 0
        assert result["limit"] == 100

    def test_kg_timeline_paginates(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_timeline

        full = tool_kg_timeline()
        total = full["total"]
        assert total == full["count"]  # seeded KG fits in one default page

        page1 = tool_kg_timeline(limit=2, offset=0)
        page2 = tool_kg_timeline(limit=2, offset=2)
        assert page1["count"] == 2
        assert page1["total"] == total
        assert page1["offset"] == 0 and page1["limit"] == 2
        assert page2["offset"] == 2
        # pages are disjoint and in timeline order
        assert page1["timeline"] + page2["timeline"] == full["timeline"][: 2 + page2["count"]]

    def test_kg_timeline_clamps_pagination_args(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_timeline

        result = tool_kg_timeline(limit=10_000, offset=-3)
        assert result["limit"] == 100  # clamped to _MAX_RESULTS
        assert result["offset"] == 0

    def test_kg_stats(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_stats

        result = tool_kg_stats()
        assert result["entities"] >= 4

    # --- Date validation at the MCP boundary (issue #1164) ---

    def test_kg_add_rejects_invalid_valid_from(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="Alice",
            predicate="likes",
            object="coffee",
            valid_from="Jan 2025",
        )
        assert result["success"] is False
        assert "valid_from" in result["error"]
        assert "ISO-8601" in result["error"]

    def test_kg_query_rejects_invalid_as_of(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max", as_of="March 2026")
        assert "error" in result
        assert "as_of" in result["error"]

    def test_kg_invalidate_rejects_invalid_ended(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        result = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="chess",
            ended="yesterday",
        )
        assert result["success"] is False
        assert "ended" in result["error"]

    def test_kg_query_rejects_partial_iso_dates(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        # Partial ISO dates are rejected: KG queries compare TEXT dates
        # lexicographically, so "2026-01-01" <= "2026" is False, which
        # silently excludes facts. Reject at the boundary — only YYYY-MM-DD
        # produces correct results.
        for value in ("2026", "2026-03"):
            result = tool_kg_query(entity="Max", as_of=value)
            assert "error" in result, f"accepted partial date {value!r}: {result}"

        # Full ISO-8601 dates still pass.
        result = tool_kg_query(entity="Max", as_of="2026-03-15")
        assert "error" not in result, f"rejected valid date: {result}"

    def test_kg_add_accepts_datetime_valid_from(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        result = mcp_server.tool_kg_add(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-05-06T14:23:00Z",
        )

        assert result["success"] is True

        facts = kg.query_entity("Alice", direction="outgoing")
        fact = next(r for r in facts if r["predicate"] == "works_at" and r["object"] == "Acme")

        assert fact["valid_from"] == "2026-05-06T14:23:00Z"

    def test_kg_add_accepts_datetime_valid_to(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        result = mcp_server.tool_kg_add(
            "Alice",
            "worked_at",
            "OldCo",
            valid_from="2026-05-06T14:00:00Z",
            valid_to="2026-05-06T15:00:00Z",
        )

        assert result["success"] is True

        facts = kg.query_entity("Alice", direction="outgoing")
        fact = next(r for r in facts if r["predicate"] == "worked_at" and r["object"] == "OldCo")

        assert fact["valid_from"] == "2026-05-06T14:00:00Z"
        assert fact["valid_to"] == "2026-05-06T15:00:00Z"

    def test_kg_query_accepts_datetime_as_of(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-05-06T14:00:00Z",
        )

        from mempalace import mcp_server

        result = mcp_server.tool_kg_query(
            "Alice",
            as_of="2026-05-06T14:23:00Z",
            direction="outgoing",
        )

        assert "error" not in result
        assert result["as_of"] == "2026-05-06T14:23:00Z"
        assert result["count"] == 1
        assert result["facts"][0]["object"] == "Acme"

    def test_kg_query_as_of_does_not_duplicate_ended_facts(
        self, monkeypatch, config, palace_path, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-01-01",
            valid_to="2026-06-01",
        )
        result = mcp_server.tool_kg_query("Alice", as_of="2026-04-01", direction="outgoing")
        assert result["count"] == 1
        assert len(result["active_facts"]) == 1
        assert result["historical_facts"] == []

    def test_kg_query_excludes_future_valid_from_from_active(
        self, monkeypatch, config, palace_path, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        kg.add_triple("Alice", "starts", "school", valid_from="2099-01-01")
        kg.add_triple("Alice", "lives_in", "Town", valid_from="2020-01-01")
        result = mcp_server.tool_kg_query("Alice", direction="outgoing")
        active_preds = {r["predicate"] for r in result["active_facts"]}
        future_preds = {r["predicate"] for r in result["future_facts"]}
        assert "lives_in" in active_preds
        assert "starts" not in active_preds
        assert "starts" in future_preds

    def test_kg_query_keeps_bounded_future_end_as_active(
        self, monkeypatch, config, palace_path, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        kg.add_triple(
            "Alice",
            "on_med",
            "Empagliflozin",
            valid_from="2025-01-01",
            valid_to="2099-12-31",
        )
        kg.add_triple(
            "Alice",
            "on_med",
            "Metformin",
            valid_from="2020-01-01",
            valid_to="2025-01-01",
        )
        result = mcp_server.tool_kg_query("Alice", direction="outgoing")
        active_objs = {r["object"] for r in result["active_facts"]}
        historical_objs = {r["object"] for r in result["historical_facts"]}
        assert "Empagliflozin" in active_objs
        assert "Empagliflozin" not in historical_objs
        assert "Metformin" in historical_objs

    def test_kg_invalidate_accepts_datetime_ended(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-05-06T14:00:00Z",
        )

        from mempalace import mcp_server

        result = mcp_server.tool_kg_invalidate(
            "Alice",
            "works_at",
            "Acme",
            ended="2026-05-06T14:23:00Z",
        )

        assert result["success"] is True
        assert result["ended"] == "2026-05-06T14:23:00Z"

        facts = kg.query_entity("Alice", direction="outgoing")
        fact = next(r for r in facts if r["predicate"] == "works_at" and r["object"] == "Acme")

        assert fact["valid_to"] == "2026-05-06T14:23:00Z"

    def test_kg_add_rejects_non_canonical_datetimes(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        invalid_values = [
            "2026-05-06T14:23:00+02:00",
            "2026-05-06T14:23:00-05:30",
            "2026-05-06T14:23:00.123Z",
            "2026-05-06 14:23:00",
            "2026-05-06T14:23:00",
        ]

        for value in invalid_values:
            result = mcp_server.tool_kg_add(
                "Alice",
                "works_at",
                "Acme",
                valid_from=value,
            )

            assert result["success"] is False, value
            assert "valid_from" in result["error"]
            assert "YYYY-MM-DDTHH:MM:SSZ" in result["error"]

    def test_kg_query_rejects_non_canonical_datetime_as_of(
        self, monkeypatch, config, palace_path, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        invalid_values = [
            "2026-05-06T14:23:00+02:00",
            "2026-05-06T14:23:00-05:30",
            "2026-05-06T14:23:00.123Z",
            "2026-05-06 14:23:00",
            "2026-05-06T14:23:00",
        ]

        for value in invalid_values:
            result = mcp_server.tool_kg_query(
                "Alice",
                as_of=value,
                direction="outgoing",
            )

            assert "error" in result, value
            assert "as_of" in result["error"]
            assert "YYYY-MM-DDTHH:MM:SSZ" in result["error"]

    def test_kg_invalidate_rejects_non_canonical_ended(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-05-06T14:00:00Z",
        )

        from mempalace import mcp_server

        invalid_values = [
            "2026-05-06T14:23:00+02:00",
            "2026-05-06T14:23:00-05:30",
            "2026-05-06T14:23:00.123Z",
            "2026-05-06 14:23:00",
            "2026-05-06T14:23:00",
        ]

        for value in invalid_values:
            result = mcp_server.tool_kg_invalidate(
                "Alice",
                "works_at",
                "Acme",
                ended=value,
            )

            assert result["success"] is False, value
            assert "ended" in result["error"]
            assert "YYYY-MM-DDTHH:MM:SSZ" in result["error"]

    def test_kg_add_rejects_timezone_offset_datetime(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        result = mcp_server.tool_kg_add(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2026-05-06T14:23:00+02:00",
        )

        assert result["success"] is False
        assert "valid_from" in result["error"]
        assert "YYYY-MM-DDTHH:MM:SSZ" in result["error"]

    # ── Diary Tools ─────────────────────────────────────────────────────────

    def test_kg_invalidate_accepts_long_object(self, monkeypatch, config, palace_path, kg):
        """Invalidating a fact whose object > 128 chars must not be blocked by MAX_NAME_LENGTH.

        The object parameter in kg_invalidate is a lookup key against existing
        facts, not a new entity name.  MAX_NAME_LENGTH (128) applies to entity
        identifiers — capping lookup keys silently blocks invalidation of any
        legitimately long free-text fact written before the cap existed.
        """
        _patch_mcp_server(monkeypatch, config, kg)

        long_obj = "a" * 200  # well over the 128-char entity-name cap

        # Seed a fact with a long object directly (bypassing MCP write-path
        # validation, which intentionally caps new entity names at 128 chars).
        kg.add_triple("TestSubject", "has_note", long_obj, valid_from="2026-01-01")

        from mempalace import mcp_server

        result = mcp_server.tool_kg_invalidate(
            "TestSubject",
            "has_note",
            long_obj,
            ended="2026-06-16",
        )

        assert result["success"] is True, f"invalidate rejected long object: {result}"

    def test_kg_invalidate_long_object_write_path_unaffected(
        self, monkeypatch, config, palace_path, kg
    ):
        """Write-path object validation is unchanged by the invalidate fix.

        tool_kg_add still enforces MAX_NAME_LENGTH on the object field so that
        new entity names remain bounded.  The invalidate fix is scoped to the
        lookup path only.
        """
        _patch_mcp_server(monkeypatch, config, kg)

        from mempalace import mcp_server

        long_obj = "b" * 200

        result = mcp_server.tool_kg_add(
            "TestSubject",
            "has_note",
            long_obj,
            valid_from="2026-01-01",
        )

        # write-path still rejects long objects
        assert result["success"] is False
        assert "exceeds maximum length" in result["error"]

    # ── predicate= filter ───────────────────────────────────────────────
    #
    # project:digital-p carried 2,127 active facts (~354KB of object text) on
    # 2026-08-23, far past both the fact cap and the 50KB response cap, so an
    # unfiltered query on it could only ever return an arbitrary prefix. The
    # filter makes one slice retrievable whole. Omitting it must change
    # nothing at all for existing callers.

    def test_kg_query_default_response_shape_unchanged(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max")
        # v3.10's default shape: the interval buckets plus facts/count.
        assert set(result) == {
            "entity",
            "as_of",
            "active_facts",
            "historical_facts",
            "future_facts",
            "facts",
            "count",
        }
        assert result["count"] == 3

    def test_kg_query_single_predicate(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max", predicate="does")
        assert result["predicate"] == ["does"]
        assert {f["predicate"] for f in result["facts"]} == {"does"}
        assert result["count"] == 2

    def test_kg_query_comma_separated_predicates(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Alice", predicate="parent_of,works_at")
        assert result["predicate"] == ["parent_of", "works_at"]
        assert {f["predicate"] for f in result["facts"]} == {"parent_of", "works_at"}

    def test_kg_query_predicate_list(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Alice", predicate=["parent_of", "works_at"])
        assert {f["predicate"] for f in result["facts"]} == {"parent_of", "works_at"}

    def test_kg_query_predicate_is_normalized(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Alice", predicate="Parent Of")
        assert result["predicate"] == ["parent_of"]
        assert result["count"] == 1

    def test_kg_query_predicate_composes_with_direction_and_as_of(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        expired = tool_kg_query(
            entity="Alice", predicate="works_at", direction="outgoing", as_of="2023-06-01"
        )
        current = tool_kg_query(
            entity="Alice", predicate="works_at", direction="outgoing", as_of="2025-06-01"
        )
        assert [f["object"] for f in expired["facts"]] == ["Acme Corp"]
        assert [f["object"] for f in current["facts"]] == ["NewCo"]

        incoming = tool_kg_query(entity="Max", predicate="parent_of", direction="incoming")
        outgoing = tool_kg_query(entity="Max", predicate="parent_of", direction="outgoing")
        assert incoming["count"] == 1
        assert outgoing["count"] == 0

    def test_kg_query_rejects_bad_predicate_argument(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        assert "error" in tool_kg_query(entity="Max", predicate=17)
        assert "error" in tool_kg_query(entity="Max", predicate=["does", 17])
        assert "error" in tool_kg_query(entity="Max", predicate="  ,  ")
        assert "error" in tool_kg_query(entity="Max", predicate="../etc/passwd")

    def test_kg_query_over_cap_unfiltered_returns_histogram_and_instruction(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 2)

        result = mcp_server.tool_kg_query(entity="Max")
        assert result["truncated"] is True
        assert result["total"] == 3
        assert result["count"] == 2
        # Truncation announces itself and points at the way out.
        assert "predicate=" in result["notice"]
        assert result["predicates"] == {"does": 2, "parent_of": 1}
        assert result["predicates_distinct"] == 2
        assert "predicate=" in result["predicate_hint"]

    def test_kg_query_over_cap_filtered_says_the_slice_is_still_over(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 1)

        result = mcp_server.tool_kg_query(entity="Max", predicate="does")
        assert result["truncated"] is True
        assert result["total"] == 2
        assert "ALREADY" in result["notice"]
        assert "STILL over" in result["notice"]
        assert "does" in result["notice"]
        # Already filtered: the caller knows the slice, so no histogram noise.
        assert "predicates" not in result

    def test_kg_query_zero_match_lists_the_predicates_that_do_exist(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max", predicate="learned")
        assert result["count"] == 0
        assert result["predicates"] == {"does": 2, "parent_of": 1}
        assert "synonym" in result["predicate_hint"]

    def test_kg_query_histogram_is_capped_and_says_so(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 2)
        monkeypatch.setattr(mcp_server, "KG_PREDICATE_HISTOGRAM_LIMIT", 1)

        result = mcp_server.tool_kg_query(entity="Max")
        assert result["predicates"] == {"does": 2}
        assert result["predicates_omitted"] == 1
        assert result["predicates_distinct"] == 2
        assert "not listed" in result["predicate_hint"]

    def test_kg_query_schema_documents_the_predicate_parameter(self):
        from mempalace.mcp_server import TOOLS

        entry = TOOLS["mempalace_kg_query"]
        assert "predicate" in entry["input_schema"]["properties"]
        assert entry["input_schema"]["required"] == ["entity"]
        assert "predicate" in entry["description"]

    # ── summary= mode ───────────────────────────────────────────────────
    #
    # A hub entity's query result overflowed the cap and came back as an
    # arbitrary prefix, with no way to tell what the other rows were.
    # query_entity applies no valid_to filter unless as_of is given, so the
    # over-cap count blended dead facts with live ones: project:digital-p
    # + pending_task was 335 rows / 93 current on 2026-08-24, and the 100
    # returned held only 45 live facts. Summary mode returns a complete
    # per-predicate histogram split current/invalidated, plus a sample
    # spread across predicates and biased to live facts.

    def test_kg_summary_is_absent_when_the_result_fits(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        # The no-op guarantee: default summary= never touches a fitting response.
        assert set(tool_kg_query(entity="Max")) == {
            "entity",
            "as_of",
            "active_facts",
            "historical_facts",
            "future_facts",
            "facts",
            "count",
        }
        assert "summary" not in tool_kg_query(entity="Max", predicate="does")

    def test_kg_summary_opt_in_summarizes_an_entity_that_fits(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Alice", direction="outgoing", summary=True)
        summary = result["summary"]
        assert summary["total"] == 3
        assert summary["current"] == 2
        assert summary["invalidated"] == 1
        assert summary["histogram"] == {"works_at": [1, 1], "parent_of": [1, 0]}
        # Everything fit, so nothing is withheld and nothing claims truncation.
        assert result["count"] == 3
        assert "truncated" not in result
        assert "notice" not in result

    def test_kg_summary_is_automatic_on_overflow(self, monkeypatch, config, palace_path, seeded_kg):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 2)

        result = mcp_server.tool_kg_query(entity="Max")
        assert result["truncated"] is True
        assert result["summary"]["total"] == 3
        assert result["summary"]["predicates_distinct"] == 2
        # The old "arbitrary first N" wording is a lie about a sample.
        assert "arbitrary" not in result["notice"]
        assert "arbitrary" not in result["predicate_hint"]
        assert "sample" in result["notice"]

    def test_kg_summary_false_restores_the_plain_prefix(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 2)

        result = mcp_server.tool_kg_query(entity="Max", summary=False)
        assert "summary" not in result
        assert result["truncated"] is True
        assert "showing first 2 of 3" in result["notice"]
        assert "arbitrary" in result["predicate_hint"]

    def test_kg_summary_counts_split_current_from_invalidated(
        self, monkeypatch, config, palace_path, kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        for i in range(9):
            kg.add_triple(
                "project:hub",
                "pending_task",
                f"dead task {i}",
                valid_from="2026-01-01",
                valid_to="2026-02-01",
            )
        for i in range(3):
            kg.add_triple("project:hub", "pending_task", f"live task {i}", valid_from="2026-03-01")
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 4)

        result = mcp_server.tool_kg_query(entity="project:hub", predicate="pending_task")
        summary = result["summary"]
        # The number a caller actually wants is 3 open, not 12 rows.
        assert summary["total"] == 12
        assert summary["current"] == 3
        assert summary["invalidated"] == 9
        assert summary["histogram"] == {"pending_task": [3, 9]}
        # A single-predicate query spends the whole budget on depth, and the
        # live facts are exhausted before any dead one takes a slot.
        assert result["count"] == 4
        assert [f["valid_to"] is None for f in result["facts"]] == [True, True, True, False]
        assert "93 current" not in result["notice"]
        assert "3 current / 9 invalidated" in result["notice"]

    def test_kg_summary_spreads_the_budget_across_predicates(
        self, monkeypatch, config, palace_path, kg
    ):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        for i in range(20):
            kg.add_triple("project:hub", "learned_pattern", f"pattern {i}", valid_from="2026-01-01")
        kg.add_triple("project:hub", "is_status", "live", valid_from="2026-01-02")
        kg.add_triple("project:hub", "pinned_to", "v3", valid_from="2026-01-03")
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 6)

        result = mcp_server.tool_kg_query(entity="project:hub")
        shown = {}
        for fact in result["facts"]:
            shown[fact["predicate"]] = shown.get(fact["predicate"], 0) + 1
        # Round-robin: the two singleton predicates are not crowded out by the
        # 20-fact one, which is exactly what a prefix did to them.
        assert shown == {"learned_pattern": 4, "is_status": 1, "pinned_to": 1}
        assert result["summary"]["predicates_sampled"] == 3
        assert "predicates_unsampled" not in result["summary"]

    def test_kg_summary_histogram_lists_every_predicate(self, monkeypatch, config, palace_path, kg):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        for i in range(40):
            kg.add_triple("project:hub", f"pred_{i:02d}", f"object {i}", valid_from="2026-01-01")
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 5)

        result = mcp_server.tool_kg_query(entity="project:hub")
        summary = result["summary"]
        # The top-level 'predicates' field is still capped at 25; the summary
        # histogram is not, because "nothing invisible" is its whole job.
        assert len(result["predicates"]) == mcp_server.KG_PREDICATE_HISTOGRAM_LIMIT
        assert len(summary["histogram"]) == 40
        assert summary["predicates_distinct"] == 40
        assert summary["predicates_sampled"] == 5
        assert summary["predicates_unsampled"] == 35
        assert "histogram_omitted" not in summary

    def test_kg_summary_histogram_cap_announces_itself(self, monkeypatch, config, palace_path, kg):
        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        for i in range(10):
            kg.add_triple("project:hub", f"pred_{i:02d}", f"object {i}", valid_from="2026-01-01")
        monkeypatch.setattr(mcp_server, "KG_SUMMARY_HISTOGRAM_LIMIT", 4)

        summary = mcp_server.tool_kg_query(entity="project:hub", summary=True)["summary"]
        assert len(summary["histogram"]) == 4
        assert summary["histogram_omitted"] == 6
        assert "not listed" in summary["histogram_notice"]

    def test_kg_summary_response_stays_under_the_response_cap(
        self, monkeypatch, config, palace_path, kg
    ):
        import json

        from mempalace import mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        for i in range(300):
            kg.add_triple(
                "project:hub", f"pred_{i % 30:02d}", "x" * 120 + str(i), valid_from="2026-01-01"
            )
        monkeypatch.setattr(mcp_server, "MAX_RESPONSE_CHARS", 20000)

        result = mcp_server.tool_kg_query(entity="project:hub")
        text = json.dumps(result, indent=2, ensure_ascii=False)
        # Sized here, not chopped by the generic backstop — so the notice is
        # still true and the rarest predicates were not silently deleted.
        assert len(text) <= mcp_server.MAX_RESPONSE_CHARS
        assert result["count"] == len(result["facts"]) == result["summary"]["facts_shown"]
        assert f"showing {result['count']} of 300" in result["notice"]
        _, fitted_text = mcp_server._fit_response(result, tool_name="mempalace_kg_query")
        assert fitted_text == text

    def test_kg_summary_composes_with_predicate_direction_and_as_of(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        # as_of already filters validity in SQL, so nothing is invalidated here.
        result = tool_kg_query(
            entity="Alice", direction="outgoing", as_of="2025-06-01", summary=True
        )
        assert result["summary"]["histogram"] == {"parent_of": [1, 0], "works_at": [1, 0]}

        filtered = tool_kg_query(
            entity="Alice", direction="outgoing", predicate="works_at", summary=True
        )
        assert filtered["summary"]["histogram"] == {"works_at": [1, 1]}
        assert filtered["predicate"] == ["works_at"]

    def test_kg_summary_rejects_a_non_boolean(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        assert "error" in tool_kg_query(entity="Max", summary="yes")
        assert "error" in tool_kg_query(entity="Max", summary=1.5)

    def test_kg_summary_of_a_zero_match_query_stays_a_dead_end_rescue(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max", predicate="learned", summary=True)
        assert result["count"] == 0
        assert "summary" not in result
        assert "synonym" in result["predicate_hint"]

    def test_kg_query_schema_documents_the_summary_parameter(self):
        from mempalace.mcp_server import TOOLS

        entry = TOOLS["mempalace_kg_query"]
        assert entry["input_schema"]["properties"]["summary"]["type"] == "boolean"
        assert entry["input_schema"]["required"] == ["entity"]
        assert "summary" in entry["description"]

    def test_kg_invalidate_no_match_is_not_a_success(
        self, monkeypatch, config, palace_path, seeded_kg, caplog
    ):
        """An object that matches no open fact must not come back success-shaped.

        The object is a lookup key: a typo or a re-worded fact matches nothing
        and the UPDATE touches no rows. The old response echoed the CALLER'S
        input back as ``fact`` with ``success: true``, so a stale fact stayed
        live behind what read as confirmation of a write.
        """
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace import mcp_server

        with caplog.at_level(logging.WARNING, logger="mempalace_mcp"):
            result = mcp_server.tool_kg_invalidate(
                subject="Max",
                predicate="does",
                object="a pastime nobody ever stored",
                ended="2026-03-01",
            )

        assert result["success"] is False
        assert result["matched"] == 0
        assert result.get("error_class") == "NoMatchingFact"
        assert "no open fact" in result["error"]

        # ... and the no-op is visible in the log, not only in the return value.
        assert any(
            "kg_invalidate" in rec.getMessage() and "NoMatchingFact" in rec.getMessage()
            for rec in caplog.records
        ), [rec.getMessage() for rec in caplog.records]

        # The fact it failed to close is still open.
        facts = seeded_kg.query_entity("Max", direction="outgoing")
        assert any(f["object"] == "chess" and f["current"] for f in facts)

    def test_kg_invalidate_second_call_reports_no_match(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """Re-invalidating an already-ended fact closes nothing and says so."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace import mcp_server

        first = mcp_server.tool_kg_invalidate("Max", "does", "chess", ended="2026-03-01")
        second = mcp_server.tool_kg_invalidate("Max", "does", "chess", ended="2026-03-01")

        assert first["success"] is True and first["matched"] == 1
        assert second["success"] is False and second["matched"] == 0

    # ── v3.10 port: predicate filter vs upstream candidate resolution ──

    def test_predicate_filter_applies_to_candidate_fallback(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """v3.10 resolves an unambiguous near-miss name ('Acme' -> 'Acme Corp').
        The predicate filter must reach that fallback query too."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        hit = tool_kg_query(entity="Acme", predicate="works_at")
        assert [f["predicate"] for f in hit["facts"]] == ["works_at"]
        assert hit.get("resolved_from") == "Acme"

        miss = seeded_kg.query_entity("Acme", direction="both", predicates=["parent_of"])
        assert miss == []

    def test_filtered_miss_on_existing_entity_skips_candidates(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """An entity that HAS facts answers a filtered miss with its predicate
        histogram, not with a list of similarly named entities."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max", predicate="no_such_predicate")
        assert result["count"] == 0
        assert "candidates" not in result
        assert set(result["predicates"]) == {"does", "parent_of"}

    def test_interval_buckets_only_hold_shipped_facts(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """The fact cap runs before bucketing, so active/historical/future never
        carry a fact that 'facts' dropped."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 1)
        result = mcp_server.tool_kg_query(entity="Alice", summary=False)
        shipped = result["facts"]
        bucketed = result["active_facts"] + result["historical_facts"] + result["future_facts"]
        assert len(shipped) == 1
        assert bucketed == shipped

    def test_size_cut_keeps_facts_and_buckets_consistent(
        self, monkeypatch, config, palace_path, kg
    ):
        """Under the fact cap but over the char cap: the prefix is sized in-tool,
        so 'facts' and its bucket copies are cut together and the JSON fits."""
        import json

        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        for i in range(20):
            kg.add_triple("Big", "note", f"{i:02d} " + "x" * 100, valid_from=f"2026-01-{i + 1:02d}")
        monkeypatch.setattr(mcp_server, "MAX_RESPONSE_CHARS", 4000)
        result = mcp_server.tool_kg_query(entity="Big", summary=False)
        assert len(json.dumps(result, indent=2, ensure_ascii=False)) <= 4000
        assert 0 < result["count"] < 20
        assert result["total"] == 20 and result["truncated"] is True
        assert result["active_facts"] == result["facts"]
        assert "char" in result["notice"] and "fact cap" not in result["notice"]

    def test_summary_sample_fits_with_bucket_duplication(
        self, monkeypatch, config, palace_path, kg
    ):
        import json

        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        for i in range(30):
            kg.add_triple("Hub", f"p{i % 6}", f"{i:02d} " + "y" * 80, valid_from="2026-02-01")
        monkeypatch.setattr(mcp_server, "KG_RESULT_CAP", 10)
        monkeypatch.setattr(mcp_server, "MAX_RESPONSE_CHARS", 6000)
        result = mcp_server.tool_kg_query(entity="Hub")
        assert "summary" in result
        assert len(json.dumps(result, indent=2, ensure_ascii=False)) <= 6000
        assert len(result["summary"]["histogram"]) == 6
        bucketed = result["active_facts"] + result["historical_facts"] + result["future_facts"]
        assert sorted(map(str, bucketed)) == sorted(map(str, result["facts"]))


class TestKGLazyCache:
    """Lazy per-path KnowledgeGraph cache (issue #1136)."""

    def test_lazy_init_no_import_side_effect(self, tmp_path):
        """Importing mcp_server must not create knowledge_graph.sqlite3.

        Runs in a fresh subprocess with HOME pointed at tmp_path so the
        assertion targets a clean filesystem, independent of conftest's
        session-level HOME patch.
        """

        kg_file = tmp_path / ".mempalace" / "knowledge_graph.sqlite3"
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMPAL")}
        env["HOME"] = str(tmp_path)
        env["USERPROFILE"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, "-c", "import mempalace.mcp_server"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"import failed: {result.stderr}"
        assert not kg_file.exists(), f"import created sqlite file at {kg_file} as a side effect"

    def test_get_kg_returns_same_instance(self, tmp_path, monkeypatch):
        """Two calls with the same resolved path return the same KG."""
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_kg_by_path", {})
        monkeypatch.setattr(mcp_server, "_palace_flag_given", True)
        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path))

        kg1 = mcp_server._get_kg()
        kg2 = mcp_server._get_kg()
        assert kg1 is kg2
        assert len(mcp_server._kg_by_path) == 1

    def test_get_kg_different_paths_different_instances(self, tmp_path, monkeypatch):
        """Different palace paths map to different KG instances."""
        from mempalace import mcp_server

        tmp_a = tmp_path / "a"
        tmp_b = tmp_path / "b"
        tmp_a.mkdir()
        tmp_b.mkdir()

        monkeypatch.setattr(mcp_server, "_kg_by_path", {})
        monkeypatch.setattr(mcp_server, "_palace_flag_given", True)

        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_a))
        kg_a = mcp_server._get_kg()
        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_b))
        kg_b = mcp_server._get_kg()

        assert kg_a is not kg_b
        assert len(mcp_server._kg_by_path) == 2

    def test_multi_tenant_env_switch(self, tmp_path, monkeypatch):
        """The issue #1136 acceptance scenario.

        Rotating MEMPALACE_PALACE_PATH between MCP tool calls must route
        each call to the correct tenant's KG sqlite file.
        """
        from mempalace import mcp_server

        tmp_a = tmp_path / "tenant_a"
        tmp_b = tmp_path / "tenant_b"
        tmp_a.mkdir()
        tmp_b.mkdir()

        monkeypatch.setattr(mcp_server, "_kg_by_path", {})
        monkeypatch.setattr(mcp_server, "_palace_flag_given", True)

        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_a))
        add_result = mcp_server.tool_kg_add(
            subject="alice_secret",
            predicate="owns",
            object="repo_a",
        )
        assert add_result.get("success") is True, add_result

        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_b))
        query_b = mcp_server.tool_kg_query(entity="alice_secret")
        assert query_b.get("count", 0) == 0, f"tenant B leaked tenant A's fact: {query_b}"

        monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_a))
        query_a = mcp_server.tool_kg_query(entity="alice_secret")
        assert query_a.get("count", 0) >= 1, f"tenant A lost its own fact: {query_a}"


# ── Structured error codes + MineAlreadyRunning (#1552) ─────────────────
