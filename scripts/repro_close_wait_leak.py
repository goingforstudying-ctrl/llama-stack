#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Reproducer for the abandoned-stream CLOSE_WAIT leak (issue #6437).

Starts a minimal OpenAI-compatible streaming server on localhost, abandons
chat-completion streams after the first chunk, and counts the sockets left in
CLOSE_WAIT with ss(8). Streams routed through the ogx stream wrapper are
closed on abandon, so no CLOSE_WAIT sockets should accumulate. The --leaky
mode skips the wrapper and demonstrates the leak this fix closes.

Usage:
    uv run scripts/repro_close_wait_leak.py
    uv run scripts/repro_close_wait_leak.py --requests 10
    uv run scripts/repro_close_wait_leak.py --leaky
"""

import argparse
import asyncio
import json
import subprocess
import threading
import time
from collections.abc import AsyncGenerator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from openai import AsyncOpenAI

from ogx.providers.utils.inference.stream_utils import wrap_async_stream

CHUNK_DELAY_SECONDS = 0.3
STREAM_CHUNKS = 4


class StreamingHandler(BaseHTTPRequestHandler):
    server_version = "repro-server/1.0"

    def do_POST(self):  # noqa: N802
        # Drain the request body so the connection closes with FIN instead of
        # RST when the handler returns.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i in range(STREAM_CHUNKS):
                payload = {
                    "id": f"chatcmpl-repro-{i}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"chunk {i}"},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
                time.sleep(CHUNK_DELAY_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def start_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), StreamingHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def count_sockets(port: int) -> tuple[int, int]:
    """Return (close_wait, established) socket counts for the server port."""
    try:
        out = subprocess.run(["ss", "-tanH"], capture_output=True, text=True, timeout=10).stdout
    except FileNotFoundError:
        raise SystemExit("ss(8) not found: install iproute2 to monitor connections") from None
    close_wait = established = 0
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[4].endswith(f":{port}"):
            continue
        state = fields[0]
        if state == "CLOSE-WAIT":
            close_wait += 1
        elif state == "ESTAB":
            established += 1
    return close_wait, established


async def run_client(port: int, requests: int, leaky: bool) -> tuple[int, int]:
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="test",
        timeout=10.0,
        max_retries=0,
    )
    abandoned: list[Any] = []
    for _ in range(requests):
        stream = await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        # Keep a reference so an unclosed upstream stream cannot be garbage
        # collected (and its connection closed by the finalizer) before the
        # sockets are counted.
        abandoned.append(stream)
        if leaky:
            iterator = stream.__aiter__()
            abandoned.append(iterator)
            await iterator.__anext__()
        else:
            wrapper = cast(AsyncGenerator[Any, None], wrap_async_stream(stream))
            abandoned.append(wrapper)
            await wrapper.__anext__()
            await wrapper.aclose()
    # Give the server time to finish its stream and send FIN.
    await asyncio.sleep(2 * CHUNK_DELAY_SECONDS * STREAM_CHUNKS)
    # Count while the abandoned streams are still referenced: once this
    # function returns, garbage collection may close them and hide the leak.
    return count_sockets(port)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=8, help="streams to abandon (default: 8)")
    parser.add_argument("--port", type=int, default=18080, help="mock server port (default: 18080)")
    parser.add_argument("--leaky", action="store_true", help="skip the wrapper to demonstrate the leak")
    args = parser.parse_args()

    server = start_server(args.port)
    try:
        close_wait, established = asyncio.run(run_client(args.port, args.requests, args.leaky))
        print(f"abandoned streams: {args.requests}, wrapped: {not args.leaky}")
        print(f"sockets: CLOSE_WAIT={close_wait}, ESTABLISHED={established}")
        if args.leaky:
            if close_wait == 0:
                print("expected the leak but no CLOSE_WAIT sockets were observed;")
                print("increase --requests or CHUNK_DELAY_SECONDS")
                return 1
            print(f"leak reproduced: {close_wait} sockets stuck in CLOSE_WAIT")
            return 0
        if close_wait != 0:
            print(f"regression: {close_wait} sockets stuck in CLOSE_WAIT after wrapped streams were abandoned")
            return 1
        print("no CLOSE_WAIT sockets: the wrapper closed the upstream streams on abandon")
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
