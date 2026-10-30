"""Verify that every number in README.md comes from a source file.

Rules
  - A hidden comment `<!-- sources: path, path -->` names the source files for the README text
    that follows it, up to the next such comment. Every number must have a sources comment above it.
  - Code is skipped (fenced blocks and inline `code`: commands and identifiers, not claims), and so
    are HTML comments.
  - A number is a token not glued to a name: GPT-4, Llama-2-13B, B1, p50, module5b are names.
  - Integers must equal an integer in the sources (or the length of a JSON list/object).
    Decimals must equal a source value formatted to the same number of decimals, either as is or
    times 100 (percentages).
  - Source values: every number in the file's text, plus, for JSON, every numeric leaf and the
    length of every list and object.

This checks that each number exists in its listed sources at the displayed precision; it does
not check which field it came from.

Run from the project root:
    python scripts/check_readme_numbers.py [README.md]
Exit status 1 on any failure.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = re.compile(r"<!--\s*sources:\s*(.*?)\s*-->")
NUMBER = re.compile(r"(?<![\w.\-/:#@$=+])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\w)")
SOURCE_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def source_values(path: Path) -> tuple[set[int], list[float]]:
    text = path.read_text(encoding="utf-8")
    floats = [float(x) for x in SOURCE_NUMBER.findall(text)]
    ints = {int(f) for f in floats if f.is_integer()}
    if path.suffix == ".json":
        def walk(x):
            if isinstance(x, bool):
                return
            if isinstance(x, (int, float)):
                floats.append(float(x))
                if float(x).is_integer():
                    ints.add(int(x))
            elif isinstance(x, (list, dict)):
                ints.add(len(x))
                for v in (x.values() if isinstance(x, dict) else x):
                    walk(v)
        walk(json.loads(text))
    return ints, floats


def readme_numbers(text: str):
    """Yield (line number, token, sources) for every number in non-code, non-comment text."""
    sources, in_fence, in_comment = None, False, False
    for n, line in enumerate(text.splitlines(), 1):
        m = SOURCES.search(line)
        if m:
            sources = [s.strip() for s in m.group(1).split(",") if s.strip()]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if in_comment:
            if "-->" not in line:
                continue
            line, in_comment = line.split("-->", 1)[1], False
        line = re.sub(r"<!--.*?-->", " ", line)
        if "<!--" in line:
            line, in_comment = line.split("<!--", 1)[0], True
        line = re.sub(r"`[^`]*`", " ", line)
        for m in NUMBER.finditer(line):
            yield n, m.group(0), sources


def matches(token: str, ints: set[int], floats: list[float]) -> bool:
    plain = token.replace(",", "")
    if "." not in plain:
        return int(plain) in ints
    d = len(plain.split(".")[1])
    return any(f"{v:.{d}f}" == plain or f"{100 * v:.{d}f}" == plain for v in floats)


def main() -> int:
    readme = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    cache, failures, checked = {}, [], 0
    for line, token, sources in readme_numbers(readme.read_text(encoding="utf-8")):
        checked += 1
        if not sources:
            failures.append(f"line {line}: {token!r} has no sources comment above it")
            continue
        ints, floats = set(), []
        for s in sources:
            if s not in cache:
                path = ROOT / s
                if not path.exists():
                    failures.append(f"line {line}: source {s} does not exist")
                    cache[s] = (set(), [])
                    continue
                cache[s] = source_values(path)
            ints |= cache[s][0]
            floats += cache[s][1]
        if not matches(token, ints, floats):
            failures.append(f"line {line}: {token!r} not found in {', '.join(sources)}")
    print(f"{readme.name}: {checked} numbers checked against {len(cache)} source files")
    for f in failures:
        print("  FAIL", f)
    print("OK: every number matches its sources" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
