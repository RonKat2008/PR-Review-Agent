"""Watermarks, immutability of state transitions, and the idempotency marker."""

from __future__ import annotations

import json

import pytest

from prime_pr_review.state import (
    LANE_MERGED,
    LANE_OPEN,
    State,
    StateError,
    build_marker,
    has_marker,
    is_reviewed,
    load_state,
    mark_reviewed,
    review_key,
    save_state,
    set_merged_cursor,
)


def test_missing_state_file_is_a_cold_start_not_an_error(tmp_path):
    state = load_state(tmp_path / "absent.json")

    assert state.reviewed == {}
    assert state.merged_cursor is None


def test_round_trips_through_disk(tmp_path):
    # Arrange
    path = tmp_path / "state" / "watermark.json"
    state = mark_reviewed(State.empty(), LANE_OPEN, 7, "sha-seven")

    # Act
    save_state(state, path)
    reloaded = load_state(path)

    # Assert
    assert is_reviewed(reloaded, LANE_OPEN, 7, "sha-seven")


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "deeply" / "nested" / "watermark.json"

    save_state(State.empty(), path)

    assert path.is_file()


def test_rejects_state_file_that_is_not_an_object(tmp_path):
    path = tmp_path / "watermark.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(StateError, match="JSON object"):
        load_state(path)


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "watermark.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(StateError, match="Could not read"):
        load_state(path)


def test_rejects_reviewed_field_of_wrong_type(tmp_path):
    path = tmp_path / "watermark.json"
    path.write_text(json.dumps({"reviewed": "nope"}), encoding="utf-8")

    with pytest.raises(StateError, match="must be an object"):
        load_state(path)


def test_mark_reviewed_does_not_mutate_the_original():
    original = State.empty()

    updated = mark_reviewed(original, LANE_OPEN, 1, "sha-a")

    assert original.reviewed == {}
    assert updated.reviewed != original.reviewed


def test_new_head_sha_is_not_considered_reviewed():
    """Pushing new commits must re-open a PR for review."""
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "old-sha")

    assert is_reviewed(state, LANE_OPEN, 1, "old-sha") is True
    assert is_reviewed(state, LANE_OPEN, 1, "new-sha") is False


def test_lanes_track_the_same_pr_independently():
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "sha-a")

    assert is_reviewed(state, LANE_OPEN, 1, "sha-a") is True
    assert is_reviewed(state, LANE_MERGED, 1, "sha-a") is False


def test_review_key_rejects_unknown_lane():
    with pytest.raises(ValueError, match="Unknown lane"):
        review_key("sideways", 1)


def test_set_merged_cursor_returns_new_state():
    original = State.empty()

    updated = set_merged_cursor(original, "2026-08-07T00:00:00+00:00")

    assert original.merged_cursor is None
    assert updated.merged_cursor == "2026-08-07T00:00:00+00:00"


def test_marker_embeds_the_head_sha():
    assert "deadbeef" in build_marker("deadbeef")


def test_has_marker_detects_a_prior_comment_for_the_same_sha():
    bodies = ["unrelated chatter", f"{build_marker('sha-1')}\nreview text"]

    assert has_marker(bodies, "sha-1") is True


def test_has_marker_is_false_for_a_different_sha():
    bodies = [f"{build_marker('sha-1')}\nreview text"]

    assert has_marker(bodies, "sha-2") is False


def test_has_marker_handles_empty_and_non_list_input():
    assert has_marker([], "sha-1") is False
    assert has_marker(None, "sha-1") is False
