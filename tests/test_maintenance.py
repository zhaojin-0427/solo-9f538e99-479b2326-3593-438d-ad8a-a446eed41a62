"""End-to-end tests for maintenance windows (停机检修)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "test.db")
    with TestClient(app) as test_client:
        yield test_client


def upload(client, text: str):
    return client.post(
        "/api/previews",
        files={"file": ("day.csv", text.encode("utf-8"), "text/csv")},
    )


def publish(client, text: str) -> dict:
    response = upload(client, text)
    assert response.status_code == 201, response.text
    preview = response.json()
    confirm = client.post(f"/api/previews/{preview['preview_id']}/confirm")
    assert confirm.status_code == 200, confirm.text
    return preview


def register(client, station: str, start: str, end: str):
    return client.post(
        "/api/maintenance",
        json={"station": station, "start": start, "end": end},
    )


def day_values(client, day: str) -> dict:
    data = client.get(f"/api/days/{day}").json()
    return {(r["station"], r["hour"]): r["value"] for r in data["records"]}


def test_register_and_list_windows(client):
    assert client.get("/api/maintenance").json() == {"windows": []}

    response = register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00")
    assert response.status_code == 201, response.text
    window = response.json()
    assert window["station"] == "SY-001"
    assert window["start"] == "2026-11-13T10:00:00"
    assert window["end"] == "2026-11-13T12:00:00"
    assert window["id"]
    assert window["created_at"]

    # Seconds and a space separator are accepted and normalised.
    second = register(client, "SY-002", "2026-11-14 00:00:00", "2026-11-15T00:00:00")
    assert second.status_code == 201, second.text
    assert second.json()["start"] == "2026-11-14T00:00:00"
    assert second.json()["end"] == "2026-11-15T00:00:00"

    listing = client.get("/api/maintenance").json()["windows"]
    assert {w["id"] for w in listing} == {window["id"], second.json()["id"]}

    filtered = client.get("/api/maintenance", params={"station": "SY-001"}).json()
    assert [w["id"] for w in filtered["windows"]] == [window["id"]]


def test_register_validation_errors(client):
    bad_payloads = [
        ({"station": "", "start": "2026-11-13T10:00", "end": "2026-11-13T12:00"}, "empty station"),
        ({"station": "  ", "start": "2026-11-13T10:00", "end": "2026-11-13T12:00"}, "blank station"),
        ({"start": "2026-11-13T10:00", "end": "2026-11-13T12:00"}, "missing station"),
        ({"station": "SY-001", "start": "not-a-time", "end": "2026-11-13T12:00"}, "bad start format"),
        ({"station": "SY-001", "start": "2026-11-13T10:30", "end": "2026-11-13T12:00"}, "start not whole hour"),
        ({"station": "SY-001", "start": "2026-11-13T10:00:15", "end": "2026-11-13T12:00"}, "start has seconds"),
        ({"station": "SY-001", "start": "2026-11-13T25:00", "end": "2026-11-13T26:00"}, "hour out of range"),
        ({"station": "SY-001", "start": "2026-02-30T10:00", "end": "2026-11-13T12:00"}, "invalid calendar date"),
        ({"station": "SY-001", "start": "2026-11-13T12:00", "end": "2026-11-13T10:00"}, "end before start"),
        ({"station": "SY-001", "start": "2026-11-13T10:00", "end": "2026-11-13T10:00"}, "empty range"),
        ({"station": "SY-001", "start": "2026-11-13T10:00"}, "missing end"),
    ]
    for payload, label in bad_payloads:
        response = client.post("/api/maintenance", json=payload)
        assert response.status_code == 400, (label, response.text)
        assert response.json()["error"] == "invalid_maintenance_window", label

    # Non-JSON bodies are rejected with the same error shape.
    response = client.post(
        "/api/maintenance",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_maintenance_window"

    assert client.get("/api/maintenance").json() == {"windows": []}


def test_register_rejected_when_window_covers_published_records(client):
    publish(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n"
        "SY-001,2026-11-13T11:00:00,1,10.8\n"
        "SY-002,2026-11-13T10:00:00,1,18.0\n",
    )

    # Covers both SY-001 records: rejected, conflict hours listed in order.
    response = register(client, "SY-001", "2026-11-13T09:00", "2026-11-13T12:00")
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "maintenance_conflict"
    detail = body["detail"]
    assert detail["conflict_count"] == 2
    assert [c["timestamp"] for c in detail["conflicts"]] == [
        "2026-11-13T10:00:00",
        "2026-11-13T11:00:00",
    ]

    # Start is inclusive: a window starting exactly at a published hour fails.
    response = register(client, "SY-001", "2026-11-13T11:00", "2026-11-13T13:00")
    assert response.status_code == 409
    assert [c["timestamp"] for c in response.json()["detail"]["conflicts"]] == [
        "2026-11-13T11:00:00"
    ]

    # End is exclusive: a window ending exactly at a published hour is fine.
    assert register(client, "SY-001", "2026-11-13T08:00", "2026-11-13T10:00").status_code == 201
    # A different station is unaffected.
    assert register(client, "SY-003", "2026-11-13T09:00", "2026-11-13T12:00").status_code == 201

    windows = client.get("/api/maintenance").json()["windows"]
    assert len(windows) == 2
    # Published data untouched by rejected registrations.
    assert day_values(client, "2026-11-13") == {
        ("SY-001", "10:00"): "12.4",
        ("SY-001", "11:00"): "10.8",
        ("SY-002", "10:00"): "18",
    }


def test_upload_preview_rejected_when_pending_writes_in_window(client):
    assert register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00").status_code == 201

    response = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T09:00:00,1,9.0\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n"
        "SY-001,2026-11-13T11:00:00,2,10.8\n"
        "SY-002,2026-11-13T10:00:00,1,18.0\n",
    )
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "maintenance_blocked"
    detail = body["detail"]
    assert detail["status"] == "blocked"
    assert detail["preview_id"] is None
    assert detail["counts"]["pending"] == 0
    hits = detail["maintenance_conflicts"]
    assert {(h["station"], h["timestamp"]) for h in hits} == {
        ("SY-001", "2026-11-13T10:00:00"),
        ("SY-001", "2026-11-13T11:00:00"),
    }
    assert all(h["window_start"] == "2026-11-13T10:00:00" for h in hits)
    assert all(h["window_end"] == "2026-11-13T12:00:00" for h in hits)

    # Nothing was published and no preview was persisted.
    assert client.get("/api/days/2026-11-13").json()["count"] == 0
    assert client.get("/api/days/2026-11-13").json()["day_version"] == 0

    # The same rows outside the window upload fine.
    ok = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T09:00:00,1,9.0\n"
        "SY-001,2026-11-13T12:00:00,1,13.0\n"
        "SY-002,2026-11-13T10:00:00,1,18.0\n",
    )
    assert ok.status_code == 201, ok.text
    assert ok.json()["counts"]["pending"] == 3

    # Same station and day but outside the window: confirm publishes normally.
    confirm = client.post(f"/api/previews/{ok.json()['preview_id']}/confirm")
    assert confirm.status_code == 200, confirm.text
    assert day_values(client, "2026-11-13") == {
        ("SY-001", "09:00"): "9",
        ("SY-001", "12:00"): "13",
        ("SY-002", "10:00"): "18",
    }


def test_window_spanning_midnight_blocks_both_days(client):
    assert register(client, "SY-001", "2026-11-13T22:00", "2026-11-14T02:00").status_code == 201

    blocked = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-14T00:00:00,1,9.0\n"
        "SY-001,2026-11-14T01:00:00,1,8.0\n",
    )
    assert blocked.status_code == 409
    assert blocked.json()["error"] == "maintenance_blocked"

    ok = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-14T02:00:00,1,9.0\n",
    )
    assert ok.status_code == 201, ok.text


def test_preview_invalidated_by_window_registered_afterwards(client):
    preview = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n",
    ).json()
    assert preview["status"] == "preview"

    assert register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00").status_code == 201

    confirm = client.post(f"/api/previews/{preview['preview_id']}/confirm")
    assert confirm.status_code == 409
    assert confirm.json()["error"] == "preview_blocked"
    assert confirm.json()["detail"]["maintenance_conflicts"][0]["timestamp"] == (
        "2026-11-13T10:00:00"
    )

    # Nothing published; the preview stays blocked on reads and re-confirms.
    assert client.get("/api/days/2026-11-13").json()["count"] == 0
    fetched = client.get(f"/api/previews/{preview['preview_id']}").json()
    assert fetched["status"] == "blocked"
    assert fetched["maintenance_conflicts"]
    again = client.post(f"/api/previews/{preview['preview_id']}/confirm")
    assert again.status_code == 409
    assert again.json()["error"] == "preview_blocked"

    # Re-uploading the same rows is now rejected at preview time.
    reupload = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n",
    )
    assert reupload.status_code == 409
    assert reupload.json()["error"] == "maintenance_blocked"


def test_unrelated_previews_remain_publishable(client):
    first = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n",
    ).json()
    second = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-002,2026-11-13T10:00:00,1,18.0\n",
    ).json()
    # Same station but a day the window does not touch (the day-version guard
    # already forbids two publishes for one day, so use another day here).
    third = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-14T10:00:00,1,7.5\n",
    ).json()

    # Window covers SY-001 10:00 on 2026-11-13 only.
    assert register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00").status_code == 201

    assert client.post(f"/api/previews/{first['preview_id']}/confirm").status_code == 409
    assert client.post(f"/api/previews/{second['preview_id']}/confirm").status_code == 200
    assert client.post(f"/api/previews/{third['preview_id']}/confirm").status_code == 200

    assert day_values(client, "2026-11-13") == {("SY-002", "10:00"): "18"}
    assert day_values(client, "2026-11-14") == {("SY-001", "10:00"): "7.5"}


def test_register_and_confirm_race_is_serialized(client):
    publish(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T10:00:00,1,12.4\n",
    )
    preview = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-13T11:00:00,2,10.8\n",
    ).json()

    def do_confirm():
        return client.post(f"/api/previews/{preview['preview_id']}/confirm").status_code

    def do_register():
        return register(client, "SY-001", "2026-11-13T11:00", "2026-11-13T13:00").status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda fn: fn(), [do_confirm, do_register]))

    # Exactly one of them wins (confirm -> 200, register -> 201); the loser
    # sees the other's committed state and gets 409.
    assert sorted(statuses) in ([200, 409], [201, 409])

    windows = client.get("/api/maintenance").json()["windows"]
    published = day_values(client, "2026-11-13")
    if statuses[0] == 200:
        # Confirm committed first: the record exists and registration failed.
        assert published.get(("SY-001", "11:00")) == "10.8"
        assert windows == []
    else:
        # Registration committed first: the window exists and nothing was
        # published inside it.
        assert len(windows) == 1
        assert ("SY-001", "11:00") not in published
    # Invariant in both orderings: no published record inside an effective window.
    for window in windows:
        for (station, hour), _ in published.items():
            if station != window["station"]:
                continue
            ts = f"2026-11-13T{hour}:00"
            assert not (window["start"] <= ts < window["end"])


def test_window_does_not_block_other_days_or_stations(client):
    assert register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00").status_code == 201

    other_day = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-001,2026-11-14T10:00:00,1,12.4\n",
    )
    assert other_day.status_code == 201, other_day.text

    other_station = upload(
        client,
        "station,timestamp,revision,value\n"
        "SY-002,2026-11-13T10:00:00,1,18.0\n",
    )
    assert other_station.status_code == 201, other_station.text


def test_overlapping_windows_for_same_station_allowed(client):
    assert register(client, "SY-001", "2026-11-13T10:00", "2026-11-13T12:00").status_code == 201
    assert register(client, "SY-001", "2026-11-13T11:00", "2026-11-13T13:00").status_code == 201
    assert len(client.get("/api/maintenance").json()["windows"]) == 2
