"""Unit tests for backend/geocoding.py -- no live Nominatim calls."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend import geocoding
from backend.tests.test_api import client


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(geocoding, "CACHE_DIR", str(tmp_path))
    return tmp_path


NOMINATIM_ROWS = [
    {"display_name": "Osnabrück, Niedersachsen, Deutschland", "lat": "52.2719595", "lon": "8.047635"},
]


class TestGeocode:
    @patch("backend.geocoding.requests.get")
    def test_live_lookup_parses_and_caches(self, mock_get, tmp_cache):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: NOMINATIM_ROWS, raise_for_status=lambda: None,
        )
        out = geocoding.geocode("Osnabrück")
        assert out["source"] == "nominatim"
        assert out["results"][0]["lat"] == pytest.approx(52.2719595)
        # policy params actually sent
        params = mock_get.call_args.kwargs["params"]
        assert params["countrycodes"] == "de"
        assert "User-Agent" in mock_get.call_args.kwargs["headers"]
        # second call is served from cache without a network hit
        mock_get.reset_mock()
        again = geocoding.geocode("  osnabrück ")  # normalization: same cache key
        assert again["source"] == "cache"
        assert again["results"] == out["results"]
        mock_get.assert_not_called()

    @patch("backend.geocoding.requests.get")
    def test_no_match_is_empty_results_not_error(self, mock_get, tmp_cache):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: [], raise_for_status=lambda: None,
        )
        out = geocoding.geocode("xyzzy-no-such-place")
        assert out == {"results": [], "source": "nominatim"}

    @patch("backend.geocoding.requests.get")
    def test_network_failure_without_cache_raises(self, mock_get, tmp_cache):
        mock_get.side_effect = requests.ConnectionError("offline")
        with pytest.raises(geocoding.GeocodingUnavailable):
            geocoding.geocode("Köln")

    def test_empty_query_short_circuits(self, tmp_cache):
        assert geocoding.geocode("   ")["results"] == []

    @patch("backend.geocoding.time.sleep")
    @patch("backend.geocoding.requests.get")
    def test_rate_limit_spacing_enforced(self, mock_get, mock_sleep, tmp_cache, monkeypatch):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: [], raise_for_status=lambda: None,
        )
        monkeypatch.setattr(geocoding, "_last_request_at", 0.0)
        geocoding.geocode("first place")
        geocoding.geocode("second place")  # immediately after -> must sleep
        assert mock_sleep.called
        assert mock_sleep.call_args[0][0] > 0


class TestGeocodeEndpoint:
    @patch("backend.geocoding.geocode")
    def test_endpoint_returns_results(self, mock_geo):
        mock_geo.return_value = {"results": [{"display_name": "X", "lat": 52.0, "lon": 8.0}], "source": "cache"}
        resp = client.get("/api/geocode?q=X")
        assert resp.status_code == 200
        assert resp.json()["results"][0]["lat"] == 52.0

    @patch("backend.geocoding.geocode")
    def test_endpoint_502_when_unavailable(self, mock_geo):
        mock_geo.side_effect = geocoding.GeocodingUnavailable("offline")
        resp = client.get("/api/geocode?q=X")
        assert resp.status_code == 502
