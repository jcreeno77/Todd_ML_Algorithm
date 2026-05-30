"""Progress + ETA trailing for long-running collection jobs.

Every data job that loops over many symbols/dates and hits the rate-limited
market API (backfill, nightly collector, gap scanner) should wrap its item loop
with :func:`track` so the run is never a silent black box -- you can trail how
far it is and roughly how long is left.

Usage
-----
    for symbol in track(symbols, "backfill"):
        ...                       # one line per item, with a rolling ETA

    for cand in track(candidates, "collector:minute", key=lambda c: c["symbol"]):
        ...

Each iteration prints, to stdout (flushed so it streams through ``tee``)::

    [backfill] 7/23 NEXR (elapsed 2.1m, eta ~4.6m)

ETA is a rolling mean of completed iterations -- approximate, but enough to know
whether a rate-limited pull is minutes or tens of minutes from done.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Iterable, Iterator, TypeVar

__all__ = ["track"]

T = TypeVar("T")


def track(
    items: Iterable[T],
    label: str = "job",
    *,
    key: Callable[[T], object] = str,
    out=None,
) -> Iterator[T]:
    """Yield ``items`` while printing ``[label] i/total <item> (elapsed, eta)``.

    ``key`` renders each item's display string (default ``str``); pass e.g.
    ``key=lambda c: c["symbol"]`` for dict items. ETA is a rolling average over
    completed iterations, so it stabilises after the first item.
    """
    items = list(items)
    total = len(items)
    t0 = time.time()
    stream = out if out is not None else sys.stdout
    for i, item in enumerate(items, 1):
        done = i - 1
        elapsed = time.time() - t0
        eta = ""
        if done:
            avg = elapsed / done
            eta = f", eta ~{avg * (total - done) / 60:.1f}m"
        print(
            f"[{label}] {i}/{total} {key(item)} (elapsed {elapsed / 60:.1f}m{eta})",
            file=stream,
            flush=True,
        )
        yield item
