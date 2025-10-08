from __future__ import annotations
import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, Any, Optional

import requests


def build_url(base: str, path: str) -> str:
    base = (base or '').rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"


def build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def print_result(name: str, ok: bool, status: int, detail: str = "") -> None:
    prefix = "PASS" if ok else "FAIL"
    extra = f" | {detail}" if detail else ""
    print(f"[{prefix}] {name}: HTTP {status}{extra}")


def test_search(base_url: str, api_key: Optional[str]) -> bool:
    url = build_url(base_url, "api/search")
    params = {"q": "excavator", "k": "1"}
    try:
        resp = requests.get(url, params=params, headers=build_headers(api_key), timeout=20)
        ok = resp.status_code == 200
        detail = ""
        if ok:
            try:
                data = resp.json()
                results = data.get("results")
                detail = f"results={len(results) if isinstance(results, list) else 'n/a'}"
            except Exception:
                detail = "non-json response"
        print_result("GET /api/search", ok, resp.status_code, detail)
        return ok
    except Exception as e:
        print_result("GET /api/search", False, 0, str(e))
        return False


def test_presign_upload(base_url: str, api_key: Optional[str]) -> bool:
    url = build_url(base_url, "api/presign-upload")
    key = f"smoke-tests/{int(time.time())}-test.jpg"
    payload = {"key": key, "content_type": "image/jpeg"}
    try:
        resp = requests.post(url, data=json.dumps(payload), headers=build_headers(api_key), timeout=20)
        ok = resp.status_code == 200
        detail = ""
        if ok:
            try:
                data = resp.json()
                detail = "url returned" if data.get("url") else "no url"
            except Exception:
                detail = "non-json response"
        print_result("POST /api/presign-upload", ok, resp.status_code, detail)
        return ok
    except Exception as e:
        print_result("POST /api/presign-upload", False, 0, str(e))
        return False


def test_upsert_via_s3(base_url: str, api_key: Optional[str], s3_key: Optional[str], building: str, shot_date: str) -> bool:
    url = build_url(base_url, "api/images/upsert-s3")
    payload: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "s3_key": s3_key or f"smoke-tests/{int(time.time())}-dummy.jpg",
        "building": building,
        "shot_date": shot_date,
        "notes": "smoke-test",
    }
    try:
        resp = requests.post(url, data=json.dumps(payload), headers=build_headers(api_key), timeout=60)
        ok = resp.status_code in (200, 201)
        detail = ""
        if ok:
            try:
                data = resp.json()
                detail = f"id={data.get('id')}"
            except Exception:
                detail = "non-json response"
        print_result("POST /api/images/upsert-s3", ok, resp.status_code, detail)
        return ok
    except Exception as e:
        print_result("POST /api/images/upsert-s3", False, 0, str(e))
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for HYBRAG backend endpoints")
    parser.add_argument("--app-url", default=os.getenv("APP_URL"), help="Base URL of the deployed App Runner service")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="Optional API key header value")
    parser.add_argument("--run", nargs="*", default=["search", "presign"], choices=["search", "presign", "upsert"], help="Which tests to run")
    parser.add_argument("--s3-key", default=os.getenv("SMOKE_S3_KEY"), help="S3 key to use for upsert-s3 (optional)")
    parser.add_argument("--building", default=os.getenv("SMOKE_BUILDING", "test-building"))
    parser.add_argument("--shot-date", default=os.getenv("SMOKE_SHOT_DATE", "2025-01-01"))
    args = parser.parse_args()

    if not args.app_url:
        print("APP_URL is required (use --app-url or set env APP_URL)")
        return 2

    overall_ok = True
    if "search" in args.run:
        overall_ok = test_search(args.app_url, args.api_key) and overall_ok
    if "presign" in args.run:
        overall_ok = test_presign_upload(args.app_url, args.api_key) and overall_ok
    if "upsert" in args.run:
        overall_ok = test_upsert_via_s3(args.app_url, args.api_key, args.s3_key, args.building, args.shot_date) and overall_ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())


