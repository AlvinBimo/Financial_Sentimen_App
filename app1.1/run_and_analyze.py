from config import MODEL_BATCH_SIZE

from analyze_sentiment import load_sa_pipeline, analyze_sentiment
import streamlit as st
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datetime import date
from time import perf_counter


def run_spider_in_separate_process(
        output_file: Path,
        max_items: Optional[int] = None,
        spiders: List[str] = [],
        **kwargs
) -> None:
    """
    Call run_spider in a separate process.
    This is necessary as scrapy reactor is not restartable.
    """
    if max_items:
        function_call = f'run_spider("{output_file}", {max_items}, {spiders}, **{kwargs})'
    else:
        function_call = f'run_spider("{output_file}", spiders={spiders}, **{kwargs})'

    cmd = [
        'python3',
        '-c',
        f'from run_spider import run_spider; {function_call}'
    ]
    subprocess.run(cmd, check=True)


@st.cache_data(show_spinner=False, ttl=3600)
def get_single_spider_data(
        spider: str,
        output_file: Path,
        query: str
) -> pd.DataFrame:
    """
    Get data from single spider with query and save to output_file.
    """
    if os.path.exists(output_file) and os.path.isfile(output_file):
        os.remove(output_file)

    kwargs = {'query': query}
    run_spider_in_separate_process(output_file, spiders=[spider], **kwargs)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return pd.read_csv(output_file)
    else:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def collect_data_and_analyze_sentiment(
        output_file: Path,
        spiders: List[str],
        max_items: int,
        query: str,
        start_date: date,
        end_date: date,
        model_path: str,
        batch_size: int = MODEL_BATCH_SIZE
) -> pd.DataFrame:
    """
    Get data with specified spiders, analyze sentiment, and save to output_file.
    """
    kwargs = {}
    kwargs['query'] = query
    if start_date and end_date:
        kwargs['start_date'] = start_date.strftime('%Y-%m-%d')
        kwargs['end_date'] = end_date.strftime('%Y-%m-%d')

    if os.path.exists(output_file) and os.path.isfile(output_file):
        os.remove(output_file)

    total_start = perf_counter()
    scrape_start = perf_counter()
    run_spider_in_separate_process(output_file, max_items, spiders, **kwargs)
    scrape_end = perf_counter()


    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        sa_pipeline = load_sa_pipeline(model_path, batch_size)
        df = analyze_sentiment(output_file, sa_pipeline)
        inference_start = perf_counter()
        inference_end = perf_counter()

        total_end = perf_counter()

        timings = {
            'scrape_seconds': max(0.0, scrape_end - scrape_start),
            'inference_seconds': max(0.0, inference_end - inference_start),
            'total_seconds': max(0.0, total_end - total_start),
            'scrape_start': scrape_start,
            'scrape_end': scrape_end,
            'inference_start': inference_start,
            'inference_end': inference_end,
            'total_start': total_start,
            'total_end': total_end,
        }

        # Try to store timings into Streamlit session state for UI display. If streamlit
        # session_state is not available, ignore silently.
        try:
            st.session_state['timings'] = timings
        except Exception:
            pass
        return df
    else:
        # Scraping produced no data; still record scrape timing
        total_end = perf_counter()
        timings = {
            'scrape_seconds': max(0.0, scrape_end - scrape_start),
            'inference_seconds': 0.0,
            'total_seconds': max(0.0, total_end - total_start),
            'scrape_start': scrape_start,
            'scrape_end': scrape_end,
            'total_start': total_start,
            'total_end': total_end,
        }
        try:
            st.session_state['timings'] = timings
        except Exception:
            pass
        return pd.DataFrame()
