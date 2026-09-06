"""Fail-closed direct downloader for large public assets.

This module deliberately does not inherit the parent process environment.  It
does not implement proxy fallback, proxy configuration, or credential input.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import write_json


PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
DIRECT_CURL_FLAGS = [
    "-q",
    "--proxy", "",
    "--noproxy", "*",
    "--fail",
    "--show-error",
    "--location",
    "--connect-timeout", "15",
    "--max-redirs", "5",
    "--retry", "2",
    "--proto", "=https",
    "--proto-redir", "=https",
]
DIRECT_CURL_BIN = shutil.which("curl") or "/usr/bin/curl"


def _run_clean(argv: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    clean_env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    return subprocess.run(argv, env=clean_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)


def _command_output(argv: List[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        result = _run_clean(argv, timeout=timeout)
        return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"argv": argv, "error": type(exc).__name__ + ": " + str(exc)}


def audit_local_route(url: str) -> Dict[str, Any]:
    """Audit observable routing without reading proxy values or credentials."""
    parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"status": "BLOCKED_INVALID_HTTPS_URL", "url": url}
    host = parsed.hostname
    try:
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    except Exception as exc:
        addresses = []
        dns_error = type(exc).__name__ + ": " + str(exc)
    else:
        dns_error = None
    links = _command_output(["/sbin/ip", "-brief", "link"])
    routes = _command_output(["/sbin/ip", "route", "show", "default"])
    rules = _command_output(["/sbin/ip", "rule", "show"])
    process_names = _command_output(["/usr/bin/ps", "-eo", "comm="])
    names = set((process_names.get("stdout") or "").split())
    tunnel_links = []
    for line in (links.get("stdout") or "").splitlines():
        if re.search(r"\b(tun|tap|wg|tailscale|ppp|vpn)\b", line, flags=re.IGNORECASE):
            tunnel_links.append(line)
    proxychains_seen = sorted(n for n in names if "proxychain" in n.lower())
    route_text = (routes.get("stdout") or "") + (rules.get("stdout") or "")
    route_red_flags = [line for line in route_text.splitlines() if any(token in line.lower() for token in ("tun", "tap", "wg", "proxy", "redsocks"))]
    observed_proxy_daemons = sorted(n for n in names if n.lower() in ("mihomo", "clash", "clash-meta", "redsocks", "privoxy"))
    status = "VERIFIED_DIRECT_ROUTE"
    reason = []
    if not addresses:
        status = "BLOCKED_DNS"
        reason.append("no DNS address")
    if tunnel_links:
        status = "BLOCKED_TUNNEL_ROUTE"
        reason.append("tunnel-like link present")
    if proxychains_seen:
        status = "BLOCKED_PROXYCHAINS"
        reason.append("proxychains process present")
    if route_red_flags:
        status = "BLOCKED_SUSPICIOUS_ROUTE"
        reason.append("route/rule mentions a tunnel/proxy path")
    return {
        "status": status,
        "url": url,
        "hostname": host,
        "resolved_addresses": addresses,
        "dns_error": dns_error,
        "proxy_env_keys_present_in_parent": sorted(k for k in PROXY_KEYS if k in os.environ),
        "proxy_env_values_recorded": False,
        "tunnel_links": tunnel_links,
        "proxychains_processes": proxychains_seen,
        "observed_proxy_daemons": observed_proxy_daemons,
        "route_red_flags": route_red_flags,
        "ip_link_audit": links,
        "default_route_audit": routes,
        "ip_rule_audit": rules,
        "audit_reason": reason,
        "transparent_middlebox": "not observable from the host; this is why the explicit local route audit is required",
    }


def direct_head(url: str, audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    audit = audit or audit_local_route(url)
    if audit.get("status") != "VERIFIED_DIRECT_ROUTE":
        return {"status": "NOT_RUN", "reason": "route audit did not verify direct path", "audit": audit}
    started = time.time()
    result = _run_clean([DIRECT_CURL_BIN] + DIRECT_CURL_FLAGS + ["--head", url], timeout=45)
    return {
        "status": "HEAD_OK" if result.returncode == 0 else "HEAD_FAILED",
        "returncode": result.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stderr": result.stderr,
        "headers": result.stdout,
        "audit": audit,
    }


def direct_download(url: str, output_path: str, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    audit = audit_local_route(url)
    record: Dict[str, Any] = {"url": url, "output": str(Path(output_path).name), "audit": audit, "expected_sha256": expected_sha256}
    if audit.get("status") != "VERIFIED_DIRECT_ROUTE":
        record.update({"status": "BLOCKED_NO_VERIFIED_DIRECT_ROUTE", "downloaded": False})
        return record
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = _run_clean([DIRECT_CURL_BIN] + DIRECT_CURL_FLAGS + ["--output", str(destination), url], timeout=3600)
    record.update({"returncode": result.returncode, "elapsed_s": round(time.time() - started, 3), "stderr": result.stderr, "downloaded": result.returncode == 0})
    if result.returncode != 0:
        record["status"] = "DIRECT_DOWNLOAD_FAILED_NO_PROXY_FALLBACK"
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
        return record
    actual = __import__("hashlib").sha256(destination.read_bytes()).hexdigest()
    record["sha256"] = actual
    record["bytes"] = destination.stat().st_size
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        record["status"] = "HASH_MISMATCH"
        return record
    record["status"] = "DOWNLOADED_DIRECT_NO_PROXY"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("url")
    head = sub.add_parser("head")
    head.add_argument("url")
    get = sub.add_parser("download")
    get.add_argument("url")
    get.add_argument("output")
    get.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_local_route(args.url)
    elif args.command == "head":
        result = direct_head(args.url)
    else:
        result = direct_download(args.url, args.output, args.sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
