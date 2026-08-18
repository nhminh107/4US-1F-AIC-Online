from scripts.benchmark_trake import run_benchmark


def test_benchmark_fixture_runs_without_external_services() -> None:
    rows = run_benchmark()

    assert {"baseline", "dead_end", "dense"} <= {row["fixture"] for row in rows}
    assert all(
        row["status"] == "success"
        for row in rows
        if row["mode"] != "beam_5_no_connectivity"
    )
    assert all(
        row["num_sequences"] == 5
        for row in rows
        if row["status"] == "success"
    )
    assert all("top5_matches_exact" in row for row in rows)
    assert all("dead_end_states" in row for row in rows)
    assert any(
        row["fixture"] == "dead_end"
        and row["mode"] == "beam_5"
        and row["dead_end_states"] > 0
        for row in rows
    )
    assert any(row["fixture"] == "dense" for row in rows)
