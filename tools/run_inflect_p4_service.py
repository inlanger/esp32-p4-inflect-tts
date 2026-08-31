#!/usr/bin/env python3
"""Trigger the ESP32-P4 Inflect service and capture its binary PCM as WAV."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import serial


BEGIN_RE = re.compile(
    rb"PCM_BINARY_BEGIN samples=(\d+) bytes=(\d+) sample_rate=(\d+) "
    rb"channels=(\d+) sample_width=(\d+) fnv1a=([0-9a-fA-F]{8})"
)
END_RE = re.compile(
    rb"PCM_BINARY_END bytes=(\d+) written=(\d+) fnv1a=([0-9a-fA-F]{8}) "
    rb"passed=(true|false)"
)


class SynthesisError(RuntimeError):
    """A board inference request failed after accepting its command."""

    def __init__(
        self,
        message: str,
        *,
        frontend: dict | None = None,
        error_record: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.frontend = frontend or {}
        self.error_record = error_record or {}


def fnv1a(payload: bytes) -> int:
    value = 0x811C9DC5
    for byte in payload:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def read_exact(port: serial.Serial, byte_count: int, deadline: float) -> bytes:
    payload = bytearray()
    while len(payload) < byte_count:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"PCM stopped at {len(payload)} of {byte_count} bytes"
            )
        chunk = port.read(byte_count - len(payload))
        if chunk:
            payload.extend(chunk)
    return bytes(payload)


def read_line(port: serial.Serial, deadline: float) -> bytes:
    line = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read_until(b"\n")
        if not chunk:
            continue
        line.extend(chunk)
        if line.endswith(b"\n"):
            return bytes(line).rstrip(b"\r\n")
    raise TimeoutError("timed out waiting for a service record")


def print_record(line: bytes, quiet: bool) -> None:
    if not quiet and line:
        print(line.decode("utf-8", errors="replace"))


def retain_record(line: bytes, raw_lines: list[str] | None) -> None:
    if raw_lines is not None and line:
        raw_lines.append(line.decode("utf-8", errors="replace"))


def wait_for_ready(
    port: serial.Serial,
    timeout: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    next_probe = time.monotonic()
    while time.monotonic() < deadline:
        if time.monotonic() >= next_probe:
            port.write(b"STATUS\n")
            port.flush()
            next_probe = time.monotonic() + 1.0
        try:
            line = read_line(port, min(deadline, time.monotonic() + 0.5))
        except TimeoutError:
            continue
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if b'"type":"service_ready"' in line and b'"ready":true' in line:
            return
        if b'"stage":"service_status"' in line:
            return
    raise TimeoutError("board did not enter the Inflect command service")


def run_without_pcm(
    port: serial.Serial,
    timeout: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> dict:
    port.write(b"RUN\n")
    port.flush()
    deadline = time.monotonic() + timeout
    decoder_result: dict = {}
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if line.startswith(b'{"type":"decoder_audio"'):
            decoder_result = json.loads(line)
        if b'"type":"run_complete"' in line:
            if b'"passed":true' not in line:
                raise RuntimeError(f"warmup failed: {line!r}")
            return decoder_result
    raise TimeoutError("warmup did not complete")


def run_text_without_pcm(
    port: serial.Serial,
    text: str,
    timeout: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> tuple[dict, dict]:
    """Run the complete raw-text pipeline while keeping PCM on the board."""

    port.write(text_command(text, emit_pcm=False))
    port.flush()
    deadline = time.monotonic() + timeout
    frontend_result: dict = {}
    decoder_result: dict = {}
    error_result: dict = {}
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if line.startswith(b'{"type":"text_frontend"'):
            frontend_result = json.loads(line)
        elif line.startswith(b'{"type":"decoder_audio"'):
            decoder_result = json.loads(line)
        elif line.startswith(b'{"type":"full_pipeline_error"'):
            error_result = json.loads(line)
        elif b'"type":"command_error"' in line:
            raise SynthesisError(
                f"TEXT_RUN command failed: {line!r}",
                frontend=frontend_result,
                error_record=json.loads(line),
            )
        elif b'"type":"run_complete"' in line:
            if b'"passed":true' not in line:
                raise SynthesisError(
                    f"TEXT_RUN synthesis failed: {line!r}",
                    frontend=frontend_result,
                    error_record=error_result,
                )
            if not frontend_result or not decoder_result:
                raise ValueError("TEXT_RUN completed without intact result records")
            return frontend_result, decoder_result
    raise TimeoutError("TEXT_RUN did not complete")


def receive_pcm_frame(
    port: serial.Serial,
    begin: re.Match[bytes],
    deadline: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> tuple[dict, bytes]:
    metadata = {
        "samples": int(begin.group(1)),
        "bytes": int(begin.group(2)),
        "sample_rate": int(begin.group(3)),
        "channels": int(begin.group(4)),
        "sample_width": int(begin.group(5)),
        "fnv1a": begin.group(6).decode().lower(),
    }
    payload = read_exact(port, metadata["bytes"], deadline)
    actual_fnv = f"{fnv1a(payload):08x}"
    if actual_fnv != metadata["fnv1a"]:
        raise ValueError(
            f"host PCM checksum {actual_fnv} != board {metadata['fnv1a']}"
        )

    while time.monotonic() < deadline:
        end_line = read_line(port, deadline)
        if not end_line:
            continue
        retain_record(end_line, raw_lines)
        print_record(end_line, quiet)
        end = END_RE.fullmatch(end_line)
        if end is not None:
            if int(end.group(1)) != metadata["bytes"]:
                raise ValueError("board ended PCM with a different byte count")
            if int(end.group(2)) != metadata["bytes"] or end.group(4) != b"true":
                raise IOError("board UART did not transmit the complete PCM payload")
            if end.group(3).decode().lower() != metadata["fnv1a"]:
                raise ValueError("begin/end PCM checksums differ")
            return metadata, payload

        # A USB-UART bridge can occasionally drop the start of this advisory
        # line after a large binary write. Length plus the payload FNV above
        # remain the authoritative framing and integrity checks.
        expected_tail = f"fnv1a={metadata['fnv1a']} passed=true".encode("ascii")
        if expected_tail in end_line:
            return metadata, payload
        raise ValueError(f"invalid PCM_BINARY_END record: {end_line!r}")
    raise TimeoutError("PCM frame did not end")


def capture_pcm(
    port: serial.Serial,
    timeout: float,
    quiet: bool,
    command: bytes = b"WAV\n",
    raw_lines: list[str] | None = None,
) -> tuple[dict, dict, dict, bytes]:
    port.write(command)
    port.flush()
    deadline = time.monotonic() + timeout
    frontend_result: dict = {}
    decoder_result: dict = {}
    error_result: dict = {}
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if line.startswith(b'{"type":"text_frontend"'):
            frontend_result = json.loads(line)
            continue
        if line.startswith(b'{"type":"decoder_audio"'):
            decoder_result = json.loads(line)
            continue
        if line.startswith(b'{"type":"full_pipeline_error"'):
            error_result = json.loads(line)
            continue
        if b'"type":"command_error"' in line:
            raise SynthesisError(
                f"command failed: {line!r}",
                frontend=frontend_result,
                error_record=json.loads(line),
            )
        if b'"type":"run_complete"' in line:
            raise SynthesisError(
                f"synthesis failed before PCM output: {line!r}",
                frontend=frontend_result,
                error_record=error_result,
            )
        begin = BEGIN_RE.fullmatch(line)
        if begin is None:
            continue
        metadata, payload = receive_pcm_frame(
            port, begin, deadline, quiet, raw_lines
        )

        while time.monotonic() < deadline:
            completion = read_line(port, deadline)
            retain_record(completion, raw_lines)
            print_record(completion, quiet)
            if b'"type":"run_complete"' in completion:
                if b'"passed":true' not in completion:
                    raise RuntimeError(f"WAV request failed: {completion!r}")
                if not decoder_result or (
                    command.startswith(b"TEXT ") and not frontend_result
                ):
                    raise ValueError(
                        "WAV completed without intact frontend/decoder records"
                    )
                return frontend_result, decoder_result, metadata, payload
        break
    raise TimeoutError("WAV request did not complete")


def capture_segmented_pcm(
    port: serial.Serial,
    text: str,
    timeout: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> tuple[dict, dict, list[dict], dict, bytes]:
    port.write(long_text_command(text))
    port.flush()
    deadline = time.monotonic() + timeout
    plan: dict = {}
    completion: dict = {}
    segments: list[dict] = []
    current_segment: dict = {}
    current_decoder: dict = {}
    payloads: list[bytes] = []
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if line.startswith(b'{"type":"text_segment_plan"'):
            plan = json.loads(line)
            continue
        if line.startswith(b'{"type":"long_text_segment_start"'):
            current_segment = json.loads(line)
            current_decoder = {}
            continue
        if line.startswith(b'{"type":"decoder_audio"'):
            current_decoder = json.loads(line)
            continue
        if line.startswith(b'{"type":"command_error"'):
            raise SynthesisError(
                f"TEXT_LONG command failed: {line!r}",
                error_record=json.loads(line),
            )
        begin = BEGIN_RE.fullmatch(line)
        if begin is not None:
            metadata, payload = receive_pcm_frame(
                port, begin, deadline, quiet, raw_lines
            )
            if not current_segment or not current_decoder:
                raise ValueError("segmented PCM arrived without segment metadata")
            segments.append(
                {
                    "segment": current_segment,
                    "decoder": current_decoder,
                    "pcm": metadata,
                }
            )
            payloads.append(payload)
            continue
        if line.startswith(b'{"type":"long_text_complete"'):
            completion = json.loads(line)
            if not completion.get("passed"):
                raise SynthesisError(
                    f"TEXT_LONG synthesis failed: {line!r}",
                    error_record=completion,
                )
            if not plan or len(segments) != completion.get("segments"):
                raise ValueError("TEXT_LONG completed with incomplete segment records")
            geometries = {
                (
                    item["pcm"]["sample_rate"],
                    item["pcm"]["channels"],
                    item["pcm"]["sample_width"],
                )
                for item in segments
            }
            if len(geometries) != 1:
                raise ValueError("TEXT_LONG segments have inconsistent PCM geometry")
            payload = b"".join(payloads)
            metadata = {
                "samples": sum(item["pcm"]["samples"] for item in segments),
                "bytes": len(payload),
                "sample_rate": segments[0]["pcm"]["sample_rate"],
                "channels": segments[0]["pcm"]["channels"],
                "sample_width": segments[0]["pcm"]["sample_width"],
                "fnv1a": f"{fnv1a(payload):08x}",
            }
            if metadata["samples"] != completion.get("emitted_samples"):
                raise ValueError("TEXT_LONG board/host sample totals differ")
            return plan, completion, segments, metadata, payload
    raise TimeoutError("TEXT_LONG request did not complete")


def request_status(
    port: serial.Serial,
    timeout: float,
    quiet: bool,
    raw_lines: list[str] | None = None,
) -> dict:
    """Return the service heap record after all currently resident models."""

    port.write(b"STATUS\n")
    port.flush()
    deadline = time.monotonic() + timeout
    heap: dict = {}
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        retain_record(line, raw_lines)
        print_record(line, quiet)
        if line.startswith(b'{"type":"heap"'):
            record = json.loads(line)
            if record.get("stage") == "service_status":
                heap = record
        if b'"type":"service_ready"' in line:
            if heap:
                return heap
            # wait_for_ready() historically accepted a service_status heap record
            # before consuming its trailing ready line. Ignore that stale line and
            # queue a fresh STATUS so the requested heap/ready pair is unambiguous.
            port.write(b"STATUS\n")
            port.flush()
            continue
    raise TimeoutError("timed out waiting for service status")


def write_wav(path: Path, metadata: dict, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(metadata["channels"])
        wav_file.setsampwidth(metadata["sample_width"])
        wav_file.setframerate(metadata["sample_rate"])
        wav_file.writeframes(payload)


def text_command(value: str, *, emit_pcm: bool = True) -> bytes:
    if not value or any(character in value for character in "\r\n\0"):
        raise ValueError("text must be a non-empty single line")
    verb = "TEXT" if emit_pcm else "TEXT_RUN"
    command = f"{verb} {value}\n".encode("utf-8")
    if len(command) > 511:
        raise ValueError("UTF-8 text is too long for the board command buffer")
    return command


def long_text_command(value: str) -> bytes:
    if not value or any(character in value for character in "\r\n\0"):
        raise ValueError("text must be a non-empty single line")
    command = f"TEXT_LONG {value}\n".encode("utf-8")
    if len(command) > 511:
        raise ValueError("UTF-8 text is too long for the board command buffer")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbserial-3110")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--long-text", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.transport_retries < 1:
        parser.error("--transport-retries must be at least 1")
    if args.long_text and args.text is None:
        parser.error("--long-text requires --text")
    request_command = text_command(args.text) if args.text is not None else b"WAV\n"

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.2,
        write_timeout=2.0,
        exclusive=True,
    ) as port:
        time.sleep(0.3)
        wait_for_ready(port, min(args.timeout, 15.0), args.quiet)
        warmup_result = (
            run_without_pcm(port, args.timeout, args.quiet)
            if args.warmup
            else None
        )
        for attempt in range(1, args.transport_retries + 1):
            try:
                if args.long_text:
                    plan, completion, segments, metadata, payload = (
                        capture_segmented_pcm(
                            port,
                            args.text,
                            args.timeout,
                            args.quiet,
                        )
                    )
                    frontend_result = plan
                    decoder_result = {"completion": completion, "segments": segments}
                else:
                    frontend_result, decoder_result, metadata, payload = capture_pcm(
                        port,
                        args.timeout,
                        args.quiet,
                        request_command,
                    )
                break
            except (TimeoutError, ValueError, OSError) as error:
                if attempt == args.transport_retries:
                    raise
                print(
                    f"PCM transport attempt {attempt} failed: {error}; retrying",
                    file=sys.stderr,
                )
                time.sleep(0.5)
                port.reset_input_buffer()
                wait_for_ready(port, min(args.timeout, 30.0), args.quiet)

    expected_bytes = (
        metadata["samples"] * metadata["channels"] * metadata["sample_width"]
    )
    if expected_bytes != len(payload):
        raise ValueError(f"PCM geometry expects {expected_bytes}, got {len(payload)}")
    write_wav(args.output, metadata, payload)
    result = {
        "output": str(args.output.resolve()),
        "text": args.text,
        "long_text": args.long_text,
        "frontend": frontend_result,
        "warmup": warmup_result,
        "decoder": decoder_result,
        "pcm": metadata,
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
