"""Live trading monitor page.

Auto-refreshes the live view via ``st.fragment(run_every=...)`` (Streamlit 1.58
has no ``st.autorefresh`` and the ``streamlit-autorefresh`` package is not
installed). The fragment re-runs only the live panels on its own interval; the
cached ``_data`` readers for live state/bars use a short TTL so each refresh
picks up fresh S3 writes from the tee.

State JSON shape (best-effort; written by ``data.tee.LiveTee.update_state``):
arbitrary dict per instance with an ``updated_at`` ISO timestamp. Common keys
this page understands: ``instance_id``, ``watched_symbols``, ``positions``,
``last_signal``/``signals``, ``last_price``, ``pnl``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ML_tradingAlgo.dashboard import _data

_REFRESH_SECONDS = 5
_STALE_SECONDS = 60


def render() -> None:
    st.title("Live Trading Monitor")
    st.caption(
        f"Auto-refreshing every {_REFRESH_SECONDS}s. STALE warning if an "
        f"instance heartbeat is older than {_STALE_SECONDS}s."
    )
    _live_fragment()


@st.fragment(run_every=_REFRESH_SECONDS)
def _live_fragment() -> None:
    st.caption(f"Last refreshed: {pd.Timestamp.now(tz='UTC').isoformat()}")
    states = _data.load_live_state()

    _instances_panel(states)
    st.divider()
    _per_symbol_chart(states)
    st.divider()
    _signal_log(states)


# --------------------------------------------------------------------------- #
# active instances
# --------------------------------------------------------------------------- #
def _instances_panel(states: list) -> None:
    st.subheader("Active instances")
    if not states:
        st.info("No live instances reporting state.")
        return

    for s in states:
        iid = s.get("instance_id", "?")
        stale, age = _is_stale(s.get("updated_at"))
        header = f"Instance {iid}"
        if stale:
            header += "  STALE"
        with st.container(border=True):
            top = st.columns([2, 1, 1, 1])
            top[0].markdown(f"### {header}")
            top[1].metric("PnL", _fmt_num(s.get("pnl")))
            top[2].metric("Positions", len(_as_list(s.get("positions"))))
            top[3].metric("Heartbeat age", "n/a" if age is None else f"{age:.0f}s")

            if stale:
                st.warning(
                    f"Heartbeat is stale (> {_STALE_SECONDS}s old). "
                    f"updated_at={s.get('updated_at')}"
                )

            watched = _as_list(s.get("watched_symbols"))
            if watched:
                st.write("**Watched:** " + ", ".join(str(w) for w in watched))

            last_price = s.get("last_price")
            if last_price:
                st.write(f"**Last price:** {last_price}")

            last_signal = s.get("last_signal")
            if last_signal:
                st.write(f"**Last signal:** {last_signal}")

            positions = _as_list(s.get("positions"))
            if positions:
                st.dataframe(
                    pd.DataFrame(positions),
                    use_container_width=True,
                    hide_index=True,
                )


# --------------------------------------------------------------------------- #
# per-symbol live chart
# --------------------------------------------------------------------------- #
def _per_symbol_chart(states: list) -> None:
    st.subheader("Per-symbol live chart")
    today = dt.date.today()
    symbols = _data.list_live_symbols(today)
    if not symbols:
        st.info("No symbols with live_tick_agg bars today.")
        return

    symbol = st.selectbox("Symbol", symbols)
    bars = _data.load_minute_bars(symbol, today, source="live_tick_agg")
    if bars is None or len(bars) == 0:
        st.warning(f"No live bars for {symbol} today.")
        return

    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    fig = go.Figure(
        go.Candlestick(
            x=df["ts"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="live",
        )
    )

    levels = _levels_for_symbol(states, symbol)
    for label, value, color in (
        ("entry", levels.get("entry"), "blue"),
        ("TP", levels.get("tp"), "green"),
        ("SL", levels.get("sl"), "red"),
    ):
        if value is not None:
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )

    fig.update_layout(
        title=f"{symbol} live_tick_agg ({today})",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _levels_for_symbol(states: list, symbol: str) -> dict:
    """Pull entry/TP/SL for ``symbol`` from any instance's open positions."""
    out: dict = {}
    for s in states:
        for pos in _as_list(s.get("positions")):
            if not isinstance(pos, dict) or pos.get("symbol") != symbol:
                continue
            for key, aliases in (
                ("entry", ("entry", "entry_price", "avg_price")),
                ("tp", ("tp", "take_profit", "target")),
                ("sl", ("sl", "stop_loss", "stop")),
            ):
                for a in aliases:
                    if a in pos and pos[a] is not None:
                        out[key] = _to_float(pos[a])
                        break
    return out


# --------------------------------------------------------------------------- #
# signal log
# --------------------------------------------------------------------------- #
def _signal_log(states: list) -> None:
    st.subheader("Signal log")
    rows: list = []
    for s in states:
        iid = s.get("instance_id", "?")
        signals = s.get("signals")
        if isinstance(signals, list):
            for sig in signals:
                rows.append(_signal_row(iid, sig))
        last = s.get("last_signal")
        if last is not None:
            rows.append(_signal_row(iid, last))

    rows = [r for r in rows if r]
    if not rows:
        st.info("No signals reported.")
        return

    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df = df.sort_values("ts", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _signal_row(instance_id, sig) -> dict:
    if isinstance(sig, dict):
        row = {"instance_id": instance_id}
        row.update(sig)
        return row
    return {"instance_id": instance_id, "signal": sig}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_stale(updated_at):
    if not updated_at:
        return True, None
    try:
        ts = pd.Timestamp(updated_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
        return age > _STALE_SECONDS, age
    except Exception:
        return True, None


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fmt_num(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
