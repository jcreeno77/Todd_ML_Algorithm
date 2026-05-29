"""Streamlit dashboard for the momentum-trading data pipeline.

Two pages (see :mod:`app`):

* ``corpus`` -- explore the ingested event/bar corpus and data quality.
* ``live``   -- monitor live trading instances in near-real-time.

All S3/IO logic lives in :mod:`_data` (the only unit-tested part). The page
modules (:mod:`corpus`, :mod:`live`) and :mod:`app` are Streamlit UI, verified
manually.
"""
