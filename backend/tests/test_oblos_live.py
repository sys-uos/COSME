"""Unit tests for backend/models/oblos_live.py.

Overpass-dependent logic is tested against small synthetic fixtures rather than the live public
API, which is flaky under repeated testing (an implicit rate limit -- the same reason the
original website waits 60s between requests). `fetch_route()` is exercised live against the
public OSRM demo server, which is reliable enough for that.
"""
import math

import pytest

from backend.models import oblos_live
from backend.models.oblos_live import (
    DEV_PUBLIC_OSRM_URL,
    OverpassWay,
    RouteSegment,
    _build_overpass_query,
    _geodesic_length_m,
    _haversine_m,
    _resolve_width,
    compute_obstacle_loss,
    fetch_route,
    segments_to_trace,
    simulate_route,
)


class TestGeodesic:
    def test_haversine_known_distance(self):
        # Osnabrück (52.2799, 8.0472) to Hannover (52.3759, 9.7320) ~ 120km great-circle
        d = _haversine_m(8.0472, 52.2799, 9.7320, 52.3759)
        assert 110_000 < d < 130_000

    def test_haversine_zero_for_same_point(self):
        assert _haversine_m(8.0, 52.0, 8.0, 52.0) == pytest.approx(0.0, abs=1e-6)

    def test_geodesic_length_sums_segments(self):
        coords = [(8.0, 52.0), (8.01, 52.0), (8.02, 52.0)]
        total = _geodesic_length_m(coords)
        half = _haversine_m(*coords[0], *coords[1])
        assert total == pytest.approx(2 * half, rel=1e-6)


class TestOverpassQuery:
    def test_query_splits_even_odd_nodes(self):
        node_ids = [100, 200, 300, 400, 500]
        q = _build_overpass_query(node_ids)
        assert "node(id:100,300,500)" in q
        assert "node(id:200,400)" in q

    def test_query_contains_expected_structure(self):
        q = _build_overpass_query([1, 2, 3])
        assert "way(bn)[highway]->.route_a;" in q
        assert "way.route[tunnel]->.tunnels;" in q
        assert 'way(around.route:0)[bridge][man_made!="bridge"]->.bridges;' in q
        assert ".crossing out ids geom;" in q

    def test_query_coerces_int_node_ids(self):
        # Regression test: OSRM's demo server returns some node ids as JSON
        # floats (e.g. 11434944250.0); Overpass QL rejects a trailing ".0".
        q = _build_overpass_query([1, 2])
        assert ".0" not in q


class TestWidthFallback:
    def test_bast_width_used_when_available_and_unused(self):
        data = {"1": {"bast_width": 10.0, "bwnr_tbwnr": "A1", "osm_width": 5.0, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 10.0

    def test_bast_width_is_zero_contribution_if_bwnr_already_used(self):
        # Exclusive gate (obstacle-data-processing.tsx:97-105): once a
        # bwnr_tbwnr has been counted for one crossing, ANOTHER crossing on
        # the same physical bridge contributes nothing -- it must NOT fall
        # through to osm/est/nn, or the same bridge gets double-counted.
        data = {"1": {"bast_width": 10.0, "bwnr_tbwnr": "A1", "osm_width": 5.0, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids={"A1"}) is None

    def test_bast_width_present_without_bwnr_falls_back_to_osm(self):
        # Missing bwnr_tbwnr means there's no BAST record to gate on at all --
        # the osm->est->nn waterfall applies, per the tsx's `else` branch.
        data = {"1": {"bast_width": 10.0, "bwnr_tbwnr": None, "osm_width": 5.0, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 5.0

    def test_falls_back_to_osm_width(self):
        data = {"1": {"bast_width": None, "bwnr_tbwnr": None, "osm_width": 5.0, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 5.0

    def test_falls_back_to_est_width(self):
        data = {"1": {"bast_width": None, "bwnr_tbwnr": None, "osm_width": None, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 4.0

    def test_falls_back_to_nn_width(self):
        data = {"1": {"bast_width": None, "bwnr_tbwnr": None, "osm_width": None, "est_width": None, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 3.0

    def test_missing_way_id_returns_none(self):
        assert _resolve_width(999, {}, used_bast_ids=set()) is None

    def test_bast_width_of_zero_falls_back_to_osm(self):
        # obstacle-data-processing.tsx:97 gates on JS truthiness
        # (`if (crossing.bast_width && crossing.bwnr_tbwnr)`), not a null-check --
        # a literal 0 is falsy in JS and must fall through to osm/est/nn, not be
        # treated as "valid BAST data worth zero". Moot for the real
        # obstacle_data.json (no bast_width==0 entries exist) but must still
        # match the original's semantics exactly.
        data = {"1": {"bast_width": 0, "bwnr_tbwnr": "A1", "osm_width": 5.0, "est_width": 4.0, "nn_width": 3.0}}
        assert _resolve_width(1, data, used_bast_ids=set()) == 5.0


class TestComputeObstacleLoss:
    def _segment(self, lon0, lat0, lon1, lat1, speed=10.0):
        return RouteSegment(
            index=0, begin_node=1, end_node=2, begin_coord=(lon0, lat0), end_coord=(lon1, lat1),
            distance_m=100.0, duration_s=10.0, speed_mps=speed,
            distance_along_route_m=0.0, duration_along_route_s=0.0,
        )

    def test_crossing_adds_width_over_speed(self):
        # A route segment along the equator; a crossing perpendicular to it at the midpoint.
        seg = self._segment(8.0, 52.0, 8.01, 52.0, speed=10.0)
        crossing = OverpassWay(id=42, geometry=[(8.005, 51.999), (8.005, 52.001)])
        obstacle_data = {"42": {"bast_width": 20.0, "bwnr_tbwnr": "X", "osm_width": None, "est_width": None, "nn_width": None}}
        segments = compute_obstacle_loss([seg], tunnels=[], crossings=[crossing], obstacle_data=obstacle_data)
        assert segments[0].time_under_obstruction_s == pytest.approx(20.0 / 10.0)

    def test_tunnel_adds_own_geometry_length_over_speed(self):
        seg = self._segment(8.0, 52.0, 8.01, 52.0, speed=10.0)
        tunnel_coords = [(8.003, 52.0), (8.007, 52.0)]
        tunnel = OverpassWay(id=7, geometry=tunnel_coords)
        segments = compute_obstacle_loss([seg], tunnels=[tunnel], crossings=[], obstacle_data={})
        expected_length = _geodesic_length_m(tunnel_coords)
        assert segments[0].time_under_obstruction_s == pytest.approx(expected_length / 10.0, rel=1e-6)

    def test_non_intersecting_crossing_adds_nothing(self):
        seg = self._segment(8.0, 52.0, 8.01, 52.0, speed=10.0)
        far_crossing = OverpassWay(id=1, geometry=[(9.0, 53.0), (9.01, 53.0)])
        obstacle_data = {"1": {"bast_width": 20.0, "bwnr_tbwnr": "Y", "osm_width": None, "est_width": None, "nn_width": None}}
        segments = compute_obstacle_loss([seg], tunnels=[], crossings=[far_crossing], obstacle_data=obstacle_data)
        assert segments[0].time_under_obstruction_s == 0.0

    def test_zero_speed_segment_skipped_without_error(self):
        seg = self._segment(8.0, 52.0, 8.01, 52.0, speed=0.0)
        crossing = OverpassWay(id=42, geometry=[(8.005, 51.999), (8.005, 52.001)])
        obstacle_data = {"42": {"bast_width": 20.0, "bwnr_tbwnr": "X", "osm_width": None, "est_width": None, "nn_width": None}}
        segments = compute_obstacle_loss([seg], tunnels=[], crossings=[crossing], obstacle_data=obstacle_data)
        assert segments[0].time_under_obstruction_s == 0.0

    def test_missing_width_data_skips_crossing(self):
        seg = self._segment(8.0, 52.0, 8.01, 52.0, speed=10.0)
        crossing = OverpassWay(id=999, geometry=[(8.005, 51.999), (8.005, 52.001)])
        segments = compute_obstacle_loss([seg], tunnels=[], crossings=[crossing], obstacle_data={})
        assert segments[0].time_under_obstruction_s == 0.0


class TestSegmentsToTrace:
    def test_only_positive_loss_segments_included(self):
        segments = [
            RouteSegment(0, 1, 2, (8.1, 52.1), (0, 0), 0, 0, 1, distance_along_route_m=0, duration_along_route_s=5.0, time_under_obstruction_s=2.0),
            RouteSegment(1, 2, 3, (0, 0), (0, 0), 0, 0, 1, distance_along_route_m=0, duration_along_route_s=10.0, time_under_obstruction_s=0.0),
        ]
        df = segments_to_trace(segments)
        assert len(df) == 1
        assert df.iloc[0]["timestamp"] == 5.0
        assert df.iloc[0]["lossTime"] == 2.0
        # lat/lon come from begin_coord=(lon, lat)=(8.1, 52.1) -- flipped to (lat, lon).
        assert df.iloc[0]["lat"] == 52.1
        assert df.iloc[0]["lon"] == 8.1
        assert list(df.columns) == ["timestamp", "lossTime", "lat", "lon"]

    def test_empty_segments_yields_empty_frame_with_correct_columns(self):
        df = segments_to_trace([])
        assert df.empty
        assert list(df.columns) == ["timestamp", "lossTime", "lat", "lon"]


class TestFetchRouteLive:
    """Exercised live against the public OSRM demo server."""

    def test_fetch_route_returns_segments_and_geometry(self):
        route = fetch_route(52.2799, 8.0472, 52.2700, 8.0600, osrm_url=DEV_PUBLIC_OSRM_URL)
        assert len(route.segments) > 0
        assert len(route.geometry) == len(route.segments) + 1
        assert all(isinstance(n, int) for n in route.node_ids)

    def test_cumulative_distance_along_route_is_monotonic(self):
        route = fetch_route(52.2799, 8.0472, 52.2700, 8.0600, osrm_url=DEV_PUBLIC_OSRM_URL)
        cum = [s.distance_along_route_m for s in route.segments]
        assert cum == sorted(cum)


class TestSimulateRouteCaching:
    def test_cache_round_trip(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_fetch_route(*a, **k):
            calls["n"] += 1
            seg = RouteSegment(0, 1, 2, (8.0, 52.0), (8.01, 52.0), 100.0, 10.0, 10.0, 0.0, 0.0, time_under_obstruction_s=1.5)
            return oblos_live.RouteResult(segments=[seg], geometry=[(8.0, 52.0), (8.01, 52.0)], node_ids=[1, 2])

        def fake_fetch_overpass(*a, **k):
            return [], []

        monkeypatch.setattr(oblos_live, "fetch_route", fake_fetch_route)
        monkeypatch.setattr(oblos_live, "fetch_overpass_obstacles", fake_fetch_overpass)
        monkeypatch.setattr(oblos_live, "compute_obstacle_loss", lambda segs, t, c, obstacle_data=None: segs)

        cache_dir = str(tmp_path)
        result1 = simulate_route(52.0, 8.0, 52.1, 8.1, cache_dir=cache_dir, use_cache=True)
        result2 = simulate_route(52.0, 8.0, 52.1, 8.1, cache_dir=cache_dir, use_cache=True)

        assert calls["n"] == 1  # second call hit the cache, didn't re-fetch
        assert result1["trace"].equals(result2["trace"])
        assert result1["polyline"] == result2["polyline"]

    def test_overpass_query_excludes_routes_final_node(self, tmp_path, monkeypatch):
        # getOverpassObstacleQuery builds its node lists from
        # `segments.map(s => s.begin_node)` -- one entry per segment, which
        # excludes the route's very last node (it's only ever an end_node).
        # `route.node_ids` is the full N+1-node list; simulate_route must pass
        # node_ids[:-1] to fetch_overpass_obstacles, not the full list.
        seen = {}

        def fake_fetch_route(*a, **k):
            seg = RouteSegment(0, 1, 2, (8.0, 52.0), (8.01, 52.0), 100.0, 10.0, 10.0, 0.0, 0.0)
            return oblos_live.RouteResult(segments=[seg], geometry=[(8.0, 52.0), (8.01, 52.0)],
                                           node_ids=[1, 2, 3])

        def fake_fetch_overpass(node_ids, **k):
            seen["node_ids"] = node_ids
            return [], []

        monkeypatch.setattr(oblos_live, "fetch_route", fake_fetch_route)
        monkeypatch.setattr(oblos_live, "fetch_overpass_obstacles", fake_fetch_overpass)
        monkeypatch.setattr(oblos_live, "compute_obstacle_loss", lambda segs, t, c, obstacle_data=None: segs)

        simulate_route(52.0, 8.0, 52.1, 8.1, cache_dir=str(tmp_path), use_cache=False)
        assert seen["node_ids"] == [1, 2]  # node 3 (the route's final node) excluded
