"""Corpus / ingestion explorer page.

Renders coverage metrics, a filterable event browser with a per-event
candlestick, a fundamentals-completeness panel (dead-feature risk), and a
data-quality panel. All S3/IO is delegated to :mod:`_data`; this module is pure
Streamlit UI.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ML_tradingAlgo.dashboard import _data

# 09:30 ET regular-session open, in minutes from midnight ET.
_OPEN_MIN = 9 * 60 + 30
_ET = "America/New_York"


def render() -> None:
    st.title("Corpus / Ingestion Explorer")

    cov = _data.coverage_summary()
    _coverage_overview(cov)

    st.divider()
    events = _data.load_events()
    _event_browser(events)

    st.divider()
    _fundamentals_panel(events)

    st.divider()
    _data_quality_panel(events)


# --------------------------------------------------------------------------- #
# coverage overview
# --------------------------------------------------------------------------- #
def _coverage_overview(cov: dict) -> None:
    st.subheader("Coverage overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", cov["n_events"])
    c2.metric("Symbols", cov["n_symbols"])
    span = "-"
    if cov["date_min"] and cov["date_max"]:
        span = f"{cov['date_min']} -> {cov['date_max']}"
    c3.metric("Date span", span)
    c4.metric("Sessions", len(cov["events_per_day"]))

    bt = cov["bars_per_table"]
    cols = st.columns(len(bt) or 1)
    for col, (table, n) in zip(cols, bt.items()):
        col.metric(f"{table} rows", f"{n:,}")

    if cov["events_per_day"]:
        s = pd.Series(cov["events_per_day"], name="events")
        s.index.name = "session_date"
        st.bar_chart(s)


# --------------------------------------------------------------------------- #
# event browser + candlestick
# --------------------------------------------------------------------------- #
def _event_browser(events: pd.DataFrame) -> None:
    st.subheader("Event browser")
    if events is None or len(events) == 0:
        st.info("No events in the corpus yet.")
        return

    df = events.copy()

    # filters
    f1, f2, f3 = st.columns(3)
    with f1:
        only_passed = st.checkbox("Passed filters only", value=False)
    with f2:
        symbols = sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df else []
        pick = st.multiselect("Symbols", symbols, default=[])
    with f3:
        min_gap = st.number_input("Min gap %", value=0.0, step=1.0) / 100.0

    if only_passed and "passed_filters" in df.columns:
        df = df[df["passed_filters"].astype(bool)]
    if pick:
        df = df[df["symbol"].isin(pick)]
    if "gap_pct" in df.columns and min_gap:
        df = df[pd.to_numeric(df["gap_pct"], errors="coerce").fillna(-1) >= min_gap]

    display_cols = [
        c
        for c in (
            "symbol",
            "session_date",
            "gap_pct",
            "rvol_at_open",
            "float_shares",
            "passed_filters",
            "filter_reasons",
        )
        if c in df.columns
    ]
    st.caption(f"{len(df)} event(s) match filters.")
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    if len(df) == 0:
        return

    # event selection -> chart + static fields
    labels = [
        f"{r['symbol']} @ {r.get('session_date')}"
        for _, r in df.iterrows()
    ]
    idx = st.selectbox(
        "Inspect event", range(len(df)), format_func=lambda i: labels[i]
    )
    row = df.iloc[idx]
    _event_detail(row)


def _event_detail(row: pd.Series) -> None:
    symbol = row["symbol"]
    session_date = _as_date(row.get("session_date"))

    left, right = st.columns([3, 1])

    with right:
        st.markdown("**Static fields**")
        for k in (
            "gap_pct",
            "rvol_at_open",
            "float_shares",
            "prior_close",
            "open_price",
            "premarket_high",
            "premarket_low",
            "premarket_volume",
            "short_interest_ratio",
            "sector_id",
            "earnings_date",
            "passed_filters",
            "filter_reasons",
        ):
            if k in row.index:
                st.write(f"**{k}**: {row[k]}")

    with left:
        bars = _data.load_minute_bars(symbol, session_date)
        if bars is None or len(bars) == 0:
            st.warning(f"No 1-minute bars stored for {symbol} on {session_date}.")
            return
        _candlestick(bars, title=f"{symbol} 1-min ({session_date})")


def _candlestick(bars: pd.DataFrame, title: str) -> None:
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    fig = go.Figure(
        go.Candlestick(
            x=df["ts"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )

    # shade the premarket region (bars strictly before 09:30 ET)
    et = df["ts"].dt.tz_convert(_ET)
    minutes = et.dt.hour * 60 + et.dt.minute
    premarket = df[minutes < _OPEN_MIN]
    if len(premarket):
        fig.add_vrect(
            x0=df["ts"].min(),
            x1=premarket["ts"].max(),
            fillcolor="LightSalmon",
            opacity=0.20,
            line_width=0,
            annotation_text="premarket",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# fundamentals completeness (dead-feature risk)
# --------------------------------------------------------------------------- #
def _fundamentals_panel(events: pd.DataFrame) -> None:
    st.subheader("Fundamentals completeness")
    st.caption(
        "Dead-feature risk: events missing these fields cannot use them as "
        "model inputs."
    )
    comp = _data.fundamentals_completeness(events)
    n = comp["n_events"]
    if n == 0:
        st.info("No events to evaluate.")
        return

    cols = st.columns(len(comp["fields"]))
    rows = []
    for col, (name, stats) in zip(cols, comp["fields"].items()):
        col.metric(
            f"{name} present",
            f"{stats['pct_present']:.0f}%",
            delta=f"-{stats['missing']} missing",
            delta_color="inverse",
        )
        rows.append(
            {"field": name, "present": stats["present"], "missing": stats["missing"]}
        )

    chart_df = pd.DataFrame(rows).set_index("field")
    st.bar_chart(chart_df)


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
def _data_quality_panel(events: pd.DataFrame) -> None:
    st.subheader("Data quality")
    if events is None or len(events) == 0:
        st.info("No events to evaluate.")
        return

    n_minute_unavailable = 0
    if "filter_reasons" in events.columns:
        reasons = events["filter_reasons"].fillna("").astype(str)
        n_minute_unavailable = int(
            reasons.str.contains("premarket_unavailable").sum()
            + reasons.str.contains("minute_unavailable").sum()
        )

    c1, c2 = st.columns(2)
    c1.metric("Events missing minute data", n_minute_unavailable)

    n_failed = 0
    if "passed_filters" in events.columns:
        n_failed = int((~events["passed_filters"].astype(bool)).sum())
    c2.metric("Events that failed filters", n_failed)

    if "filter_reasons" in events.columns:
        exploded = (
            events["filter_reasons"]
            .fillna("")
            .astype(str)
            .str.split(";")
            .explode()
            .str.strip()
        )
        exploded = exploded[exploded != ""]
        if len(exploded):
            counts = exploded.value_counts()
            st.caption("Filter-reason breakdown")
            st.bar_chart(counts)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_date(value):
    if value is None:
        return None
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return value
