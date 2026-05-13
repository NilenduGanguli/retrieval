#!/usr/bin/env python3
"""
visual_pack.py - Compress code into a screenshot-friendly text dump.

Use case:
    You have code in a locked-down environment (VDI, sandbox) where copy-paste
    out is blocked but you can take screenshots. This tool packs files into a
    dense base64-encoded blob that you can fullscreen in a terminal and
    screenshot, then hand to a vision-capable AI to reconstruct the files.

Quick start:
    # Compress and print to terminal (then fullscreen + screenshot)
    python visual_pack.py pack myproject/

    # Or just a few files
    python visual_pack.py pack file1.py file2.py utils/

    # Write to a file instead of stdout
    python visual_pack.py pack myproject/ -o packed.txt

    # Restore (also useful for verifying the round-trip locally)
    python visual_pack.py unpack -i packed.txt -d restored/

Output format (one line per row, easy to OCR from a screenshot):
    ===VP1===
    size=<N> lines=<M> sha256=<full-hash>
    01|<crc32-hex>|<base64-chunk>
    02|<crc32-hex>|<base64-chunk>
    ...
    ===END===

The per-line CRC32 lets the decoder identify exactly which lines came back
mangled from OCR, so only those lines need a clearer re-screenshot. The
header SHA256 verifies the fully-assembled archive end-to-end.

Screenshot tips for OCR reliability:
    - Use a monospace font large enough to read each character clearly. A
      ~140-180 column window with readable text beats a tiny-font 300-col
      window every time. Per-line CRC catches mistakes either way.
    - Fullscreen the terminal (F11 or Cmd-Ctrl-F) then screenshot the window.
    - If the dump spans multiple screens, scroll and screenshot each one.
      Line numbers identify each row so chunks reassemble in any order, even
      with overlap between screenshots.
    - PNG screenshots only - JPEG compression destroys dense text.
    - One file at a time helps too (`python visual_pack.py pack ingest.py`)
      so a single bad screenshot doesn't block other files.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import lzma
import re
import sys
import tarfile
import zlib
from pathlib import Path

EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", ".next", "dist",
    "build", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
    "target", ".cache", ".gradle", ".terraform", "coverage", ".tox",
}
BINARY_EXTS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".o", ".a", ".lib", ".exe",
    ".bin", ".class", ".jar", ".war", ".ear", ".zip", ".tar", ".gz", ".tgz",
    ".bz2", ".xz", ".7z", ".rar", ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".ico", ".webp", ".svg", ".pdf", ".mp3", ".mp4", ".avi", ".mov", ".webm",
    ".wasm", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".psd", ".sqlite",
    ".db",
}

MAGIC = "===VP1==="
END_MARKER = "===END==="


def collect(paths, max_file=2_000_000):
    """Walk paths, return {posix_relative_path: bytes}."""
    files = {}
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"warn: {p} not found", file=sys.stderr)
            continue
        if path.is_file():
            files[path.name] = path.read_bytes()
            continue
        base = path.parent
        for f in sorted(path.rglob("*")):
            if not f.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            if f.suffix.lower() in BINARY_EXTS:
                continue
            try:
                data = f.read_bytes()
            except OSError as e:
                print(f"warn: cannot read {f}: {e}", file=sys.stderr)
                continue
            if len(data) > max_file:
                print(f"skip: {f} ({len(data)}B > {max_file})", file=sys.stderr)
                continue
            files[f.relative_to(base).as_posix()] = data
    return files


def make_tarball(files):
    """Deterministic USTAR tar then LZMA preset 9 extreme."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name=name)
            info.size = len(files[name])
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(files[name]))
    return lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME)


def encode_to_text(data, line_width=110):
    """Render bytes as numbered base64 lines with per-line CRC32 (hex)."""
    b64 = base64.b64encode(data).decode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    lines = [b64[i:i + line_width] for i in range(0, len(b64), line_width)]
    width = max(2, len(str(len(lines))))
    out = [MAGIC, f"size={len(data)} lines={len(lines)} sha256={digest}"]
    for i, line in enumerate(lines, 1):
        crc = zlib.crc32(line.encode("ascii")) & 0xFFFFFFFF
        out.append(f"{i:0{width}d}|{crc:08x}|{line}")
    out.append(END_MARKER)
    return "\n".join(out)


def decode_from_text(text):
    """Parse encoded text back to bytes; verify per-line CRC and final SHA256."""
    hm = re.search(r"sha256=([a-f0-9]{64})", text)
    expected = hm.group(1) if hm else None
    parts = []          # (line_num, base64_chunk)
    bad_lines = []      # line numbers that failed CRC
    for raw in text.splitlines():
        # New format: NN|CRC32|base64
        m = re.match(r"^\s*(\d+)\s*\|([a-fA-F0-9]{8})\|([A-Za-z0-9+/=]+)\s*$", raw)
        if m:
            num, crc_hex, b64 = int(m.group(1)), m.group(2).lower(), m.group(3)
            actual = f"{zlib.crc32(b64.encode('ascii')) & 0xFFFFFFFF:08x}"
            if actual != crc_hex:
                bad_lines.append(num)
            parts.append((num, b64))
            continue
        # Legacy format: NN|base64
        m = re.match(r"^\s*(\d+)\s*\|([A-Za-z0-9+/=]+)\s*$", raw)
        if m:
            parts.append((int(m.group(1)), m.group(2)))
    if not parts:
        for raw in text.splitlines():
            s = raw.strip()
            if re.fullmatch(r"[A-Za-z0-9+/=]+", s) and len(s) >= 20:
                parts.append((len(parts) + 1, s))
    if not parts:
        raise ValueError("no encoded data lines found")
    if bad_lines:
        print(f"warn: CRC mismatch on lines {sorted(bad_lines)}", file=sys.stderr)
    parts.sort(key=lambda x: x[0])
    nums = [p[0] for p in parts]
    missing = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
    if missing:
        print(f"warn: missing lines {missing}", file=sys.stderr)
    b64 = "".join(p[1] for p in parts)
    b64 += "=" * (-len(b64) % 4)
    data = base64.b64decode(b64)
    if expected:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            print(
                f"warn: sha256 mismatch got={actual[:16]} expected={expected[:16]}",
                file=sys.stderr,
            )
    return data


def restore(archive, dest="."):
    """Decompress + extract tarball into dest, with path-traversal guard."""
    dest_p = Path(dest).resolve()
    dest_p.mkdir(parents=True, exist_ok=True)
    names = []
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive)), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = (dest_p / member.name).resolve()
            if not str(target).startswith(str(dest_p)):
                print(f"warn: refusing to extract outside dest: {member.name}", file=sys.stderr)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                target.write_bytes(src.read())
            names.append(member.name)
    return names


def cmd_pack(args):
    files = collect(args.paths, max_file=args.max_file)
    if not files:
        sys.exit("no files collected")
    src_total = sum(len(d) for d in files.values())
    archive = make_tarball(files)
    text = encode_to_text(archive, line_width=args.line_width)
    rows = len(text.splitlines())
    est_screens = max(1, (rows + 89) // 90)
    print(
        f"# files={len(files)} src={src_total:,}B archive={len(archive):,}B "
        f"rows={rows} ~screens={est_screens} "
        f"ratio={src_total / max(1, len(archive)):.1f}x",
        file=sys.stderr,
    )
    if args.output:
        Path(args.output).write_text(text)
        print(f"# wrote {args.output}", file=sys.stderr)
    else:
        print(text)


def cmd_unpack(args):
    text = Path(args.input).read_text() if args.input else sys.stdin.read()
    archive = decode_from_text(text)
    names = restore(archive, dest=args.dest)
    for n in names:
        print(f"+ {args.dest}/{n}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="compress files into screenshot-friendly text")
    p.add_argument("paths", nargs="+", help="files or directories to pack")
    p.add_argument("-o", "--output", help="write to file (default: stdout)")
    p.add_argument("-w", "--line-width", type=int, default=110,
                   help="base64 chars per line (default: 110)")
    p.add_argument("--max-file", type=int, default=2_000_000,
                   help="skip files larger than this (default: 2 MB)")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack", help="restore files from packed text")
    p.add_argument("-i", "--input", help="read packed text from file (default: stdin)")
    p.add_argument("-d", "--dest", default=".", help="extract into this dir (default: .)")
    p.set_defaults(func=cmd_unpack)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
