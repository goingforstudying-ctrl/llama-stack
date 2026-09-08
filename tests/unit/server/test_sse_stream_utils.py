# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for ogx_api.utils.sse_stream abandonment handling."""

import asyncio

import pytest

from ogx_api.utils import create_sse_event, sse_stream


async def test_sse_stream_forwards_events() -> None:
    async def event_gen():
        yield "one"
        yield "two"

    seen = []
    async for event in sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}"):
        seen.append(event)
    assert seen == [create_sse_event("one"), create_sse_event("two")]


async def test_sse_stream_closes_inner_generator_on_aclose() -> None:
    """aclose (GeneratorExit) must propagate to the wrapped generator."""
    closed = asyncio.Event()

    async def event_gen():
        try:
            while True:
                yield "event"
                await asyncio.sleep(0)
        finally:
            closed.set()

    sse = sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}")
    assert await anext(sse) == create_sse_event("event")
    await sse.aclose()
    assert closed.is_set()


async def test_sse_stream_closes_inner_generator_on_cancellation() -> None:
    closed = asyncio.Event()
    release = asyncio.Event()

    async def event_gen():
        try:
            yield "first"
            while True:
                await release.wait()
                yield "event"
        finally:
            closed.set()

    sse = sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}")
    assert await anext(sse) == create_sse_event("first")
    task = asyncio.create_task(sse.__anext__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_sse_stream_reports_errors_as_events() -> None:
    async def event_gen():
        yield "ok"
        raise ValueError("boom")

    seen = []
    async for event in sse_stream(event_gen(), create_sse_event, lambda e: f"error: {e}"):
        seen.append(event)
    assert seen == [create_sse_event("ok"), "error: boom"]
