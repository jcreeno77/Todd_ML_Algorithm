"""Streamlit entry point for the momentum-trading data dashboard.

Launch with::

    streamlit run ML_tradingAlgo/dashboard/app.py

A sidebar radio switches between the two pages (Corpus / Live). S3 config is
read from the environment by the underlying store/`_data` layer (``S3_BUCKET``,
``S3_PREFIX``, optional ``AWS_ENDPOINT_URL``).

Importing this module must be side-effect free (so it can be imported in tests /
linters). All Streamlit UI runs inside :func:`main`, which is only invoked under
a Streamlit runtime (or ``__main__``).
"""

from __future__ import annotations

import os

import streamlit as st

from ML_tradingAlgo.dashboard import corpus, live

PAGES = {
    "Corpus": corpus.render,
    "Live": live.render,
}


def _config_caption() -> str:
    bucket = os.environ.get("S3_BUCKET", "<unset>")
    prefix = os.environ.get("S3_PREFIX", "")
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    parts = [f"s3://{bucket}/{prefix}".rstrip("/")]
    if endpoint:
        parts.append(f"endpoint={endpoint}")
    return "  |  ".join(parts)


def main() -> None:
    st.set_page_config(
        page_title="Momentum Trader Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.sidebar.title("Momentum Trader")
    st.sidebar.caption(_config_caption())
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    if not os.environ.get("S3_BUCKET"):
        st.sidebar.error("S3_BUCKET is not set; data readers will fail.")
    PAGES[choice]()


def _running_under_streamlit() -> bool:
    """True when executed inside a Streamlit script run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" or _running_under_streamlit():
    main()
