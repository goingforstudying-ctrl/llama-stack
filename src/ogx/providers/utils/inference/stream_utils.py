# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import contextlib
import inspect
from collections.abc import AsyncIterator
from typing import Any

from ogx.log import get_logger

log = get_logger(name=__name__, category="providers::utils")


async def close_async_stream(stream: Any) -> None:
    """Best-effort close of a stream, accepting both aclose() and close().

    Closing is what releases the underlying HTTP response back to its pool,
    so every wrapper that iterates another stream must run this when it
    finishes, fails, or is abandoned by its consumer.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        result = close()
        if inspect.isawaitable(result):
            await result


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            yield item
    except Exception as e:
        log.error(f"Error in wrapped async stream: {e}")
        raise
    finally:
        await close_async_stream(stream)
