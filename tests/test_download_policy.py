import os

from cmot import download


def test_direct_env_does_not_mutate_parent_or_export_proxy():
    before = {key: value for key, value in os.environ.items() if "PROXY" in key.upper()}
    result = download._run_clean(["/usr/bin/env"])
    after = {key: value for key, value in os.environ.items() if "PROXY" in key.upper()}
    assert before == after
    assert all("_PROXY=" not in line.upper() for line in result.stdout.splitlines())


def test_curl_disables_config_and_proxy():
    flags = download.DIRECT_CURL_FLAGS
    assert flags[0] == "-q"
    assert ["--proxy", ""] == flags[1:3]
    assert ["--noproxy", "*"] == flags[3:5]


def test_invalid_route_blocks_without_download_or_fallback(tmp_path, monkeypatch):
    calls = []

    def fake_audit(url):
        return {"status": "BLOCKED_DNS", "url": url}

    def forbidden_run(*args, **kwargs):
        calls.append(args)
        raise AssertionError("direct downloader must not run after a blocked audit")

    monkeypatch.setattr(download, "audit_local_route", fake_audit)
    monkeypatch.setattr(download, "_run_clean", forbidden_run)
    result = download.direct_download("https://example.invalid/file.zip", str(tmp_path / "file.zip"))
    assert result["status"] == "BLOCKED_NO_VERIFIED_DIRECT_ROUTE"
    assert result["downloaded"] is False
    assert calls == []


def test_inventory_accepts_runtime_assets_without_source_paths_in_code(tmp_path):
    from cmot.inventory import build_inventory

    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    result = build_inventory(str(tmp_path), [("toy_asset", str(asset))])
    assert result["assets"]["toy_asset"]["exists"] is True
    assert result["environment"]["credential_values_recorded"] is False
