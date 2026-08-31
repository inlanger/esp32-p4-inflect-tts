#!/usr/bin/env python3
"""Reliably flash a large PDL3 package as independently verified chunks."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


PDL3_HEADER_SIZE = 60
PDL3_SHA256_OFFSET = 24
PDL3_SHA256_SIZE = 32
FLASH_SECTOR_SIZE = 4096


def parse_int(value: str) -> int:
    return int(value, 0)


def validate_pdl3(package_path: Path) -> int:
    data = bytearray(package_path.read_bytes())
    if len(data) < PDL3_HEADER_SIZE or data[:4] != b"PDL3":
        raise ValueError(f"{package_path} is not a PDL3 package")

    package_size = struct.unpack_from("<I", data, 20)[0]
    if package_size != len(data):
        raise ValueError(
            f"PDL3 package_size {package_size} does not match file size {len(data)}"
        )

    stored_digest = bytes(data[PDL3_SHA256_OFFSET : PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE])
    data[PDL3_SHA256_OFFSET : PDL3_SHA256_OFFSET + PDL3_SHA256_SIZE] = bytes(
        PDL3_SHA256_SIZE
    )
    computed_digest = hashlib.sha256(data).digest()
    if stored_digest != computed_digest:
        raise ValueError(
            "PDL3 SHA-256 mismatch: "
            f"stored={stored_digest.hex()} computed={computed_digest.hex()}"
        )
    return package_size


def plan_chunks(total_bytes: int, base_offset: int, chunk_bytes: int) -> list[tuple[int, int]]:
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if base_offset % FLASH_SECTOR_SIZE:
        raise ValueError("base offset must be flash-sector aligned")
    if chunk_bytes <= 0 or chunk_bytes % FLASH_SECTOR_SIZE:
        raise ValueError("chunk size must be a positive multiple of 4096")

    return [
        (base_offset + offset, min(chunk_bytes, total_bytes - offset))
        for offset in range(0, total_bytes, chunk_bytes)
    ]


def discover_esptool(explicit_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("ESPTOOL"):
        candidates.append(Path(os.environ["ESPTOOL"]).expanduser())
    if found := shutil.which("esptool"):
        candidates.append(Path(found))
    candidates.extend(
        sorted(
            Path.home().glob(".espressif/python_env/*/bin/esptool"),
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("esptool was not found; pass --esptool explicitly")


def build_command(
    esptool: Path,
    port: str,
    baud: int,
    address: int,
    chunk_path: Path,
    flash_mode: str,
    flash_freq: str,
    flash_size: str,
) -> list[str]:
    return [
        str(esptool),
        "--chip",
        "esp32p4",
        "--port",
        port,
        "--baud",
        str(baud),
        "--before",
        "default-reset",
        "--after",
        "hard-reset",
        "write-flash",
        "--flash-mode",
        flash_mode,
        "--flash-freq",
        flash_freq,
        "--flash-size",
        flash_size,
        hex(address),
        str(chunk_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--port", default="/dev/cu.usbserial-3110")
    parser.add_argument("--offset", type=parse_int, default=0x410000)
    parser.add_argument("--partition-bytes", type=parse_int, default=8 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=parse_int, default=512 * 1024)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--flash-mode", default="dio")
    parser.add_argument("--flash-freq", default="80m")
    parser.add_argument("--flash-size", default="16MB")
    parser.add_argument("--esptool")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    package_size = validate_pdl3(package)
    if package_size > args.partition_bytes:
        raise ValueError(
            f"package is {package_size} bytes, partition limit is {args.partition_bytes}"
        )
    if args.retries < 1:
        raise ValueError("retries must be at least one")

    chunks = plan_chunks(package_size, args.offset, args.chunk_bytes)
    esptool = discover_esptool(args.esptool)
    print(
        f"PDL3 verified: bytes={package_size} chunks={len(chunks)} "
        f"offset={hex(args.offset)} esptool={esptool}"
    )

    if args.dry_run:
        for index, (address, size) in enumerate(chunks):
            print(f"chunk={index} address={hex(address)} bytes={size}")
        return 0

    with package.open("rb") as source, tempfile.TemporaryDirectory(
        prefix="espdl-flash-chunks-"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        for index, (address, expected_size) in enumerate(chunks):
            payload = source.read(expected_size)
            if len(payload) != expected_size:
                raise RuntimeError(
                    f"short package read for chunk {index}: {len(payload)} != {expected_size}"
                )
            chunk_path = temporary_path / f"chunk-{index:02d}.bin"
            chunk_path.write_bytes(payload)
            command = build_command(
                esptool=esptool,
                port=args.port,
                baud=args.baud,
                address=address,
                chunk_path=chunk_path,
                flash_mode=args.flash_mode,
                flash_freq=args.flash_freq,
                flash_size=args.flash_size,
            )
            for attempt in range(1, args.retries + 1):
                print(
                    f"flash chunk={index}/{len(chunks) - 1} address={hex(address)} "
                    f"bytes={expected_size} attempt={attempt}/{args.retries}",
                    flush=True,
                )
                if subprocess.run(command, check=False).returncode == 0:
                    break
            else:
                raise RuntimeError(
                    f"chunk {index} failed after {args.retries} attempts"
                )

    print(
        "All chunks passed esptool verification. Reset the board and require "
        "the boot-time PDL3 whole-package verifier to pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
