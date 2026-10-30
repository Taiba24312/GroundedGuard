"""Module 1a: download the official RAGTruth files.

Run from the project root:
    python -m groundedguard.data.download          # skip files that already exist
    python -m groundedguard.data.download --force  # re-download

Writes:
    data/raw/ragtruth/response.jsonl
    data/raw/ragtruth/source_info.jsonl
    data/raw/ragtruth/manifest.json   (URL, sha256, line count, upstream commit)
"""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

from groundedguard.config import (
    RAGTRUTH_BRANCH,
    RAGTRUTH_FILES,
    RAGTRUTH_RAW_BASE,
    RAGTRUTH_REPO,
    RAW_DIR,
)

USER_AGENT = "GroundedGuard/0.1 (research project)"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def upstream_commit() -> str | None:
    """Commit SHA of the branch right now, so the run is traceable. Best effort."""
    url = f"https://api.github.com/repos/{RAGTRUTH_REPO}/commits/{RAGTRUTH_BRANCH}"
    try:
        return json.loads(_get(url, timeout=30))["sha"]
    except Exception as exc:  # network / rate limit: not fatal
        print(f"[warn] could not fetch upstream commit SHA ({exc}); continuing.")
        return None


def download_file(name: str, force: bool) -> dict:
    dest = RAW_DIR / name
    url = f"{RAGTRUTH_RAW_BASE}/{name}"

    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to re-download)")
    else:
        print(f"[get ] {url}")
        data = _get(url)
        # GitHub serves a tiny text pointer instead of the data if the file is in Git LFS.
        if data.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{name} is a Git LFS pointer, not the data. "
                "Clone the repo with git-lfs installed and copy dataset/ into data/raw/ragtruth/."
            )
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)  # atomic: no half-written file if interrupted

    raw = dest.read_bytes()
    n_lines = sum(1 for line in raw.splitlines() if line.strip())
    info = {
        "file": name,
        "url": url,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_nonempty_lines": n_lines,
    }
    print(f"[ok  ] {name}: {info['bytes']:,} bytes, {n_lines:,} lines, sha256={info['sha256'][:12]}...")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAGTruth.")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = [download_file(name, args.force) for name in RAGTRUTH_FILES]

    manifest = {
        "repo": RAGTRUTH_REPO,
        "branch": RAGTRUTH_BRANCH,
        "upstream_commit": upstream_commit(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()