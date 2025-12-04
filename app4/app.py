from run_and_analyze import collect_data_and_analyze_sentiment, get_single_spider_data
from config import (TEMP_DIR, SENTIMENT_FILE_PATH, INAPROC_FILE_PATH, PUTUSAN_MA_FILE_PATH, DATA_DIR, FEEDBACK_FILE_PATH,
                    NEWS_SOURCES, ADDITIONAL_SOURCES,SENTIMENT_COLORS, LOGO_PATH)

from display import (load_data, filter_data, get_period_count, get_sentiment_words, save_feedback, display_dataframe, 
                     visualize_sentiment_distribution, visualize_sentiment_over_time, visualize_wordcloud, display_articles,
                     colorize_multiselect_options, display_inaproc, display_putusan_ma)

import streamlit as st
import pandas as pd
import os, sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import ast
import re
from typing import List, Optional, Literal, Dict
from datetime import date
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from pathlib import Path
from streamlit_marquee import streamlit_marquee
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container



TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
if not os.access(TEMP_DIR, os.W_OK):
    # helpful message to user/developer
    raise RuntimeError(f"Directory {TEMP_DIR} is not writable. Run: sudo chown -R $(whoami):$(whoami) {TEMP_DIR}")

# --- App Configuration ---
st.set_page_config(
    page_title="J-SON",
    page_icon="📊",
    layout="centered"
)

# --- File Path Configurations ---
TEMP_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
if not os.path.exists(FEEDBACK_FILE_PATH):
    pd.DataFrame(columns=['article_title', 'sentence', 'original_label', 'original_score',
                          'feedback', 'corrected_label', 'timestamp']).to_csv(FEEDBACK_FILE_PATH, index=False)


st.session_state.user_feedback_count = st.session_state.get('user_feedback_count', 0)
MIN_FEEDBACK_REQUIRED = 5        


st.markdown("<h1 style='text-align: center;'>Jamkrindo Sentiment Analysis Opinion Network</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>(J-SON)</h1>", unsafe_allow_html=True)


st.write("""
Masukkan kata kunci pencarian Anda dan pilih sumber untuk mengumpulkan data dan melakukan analisis sentimen.\n
Untuk melihat data yang telah dikumpulkan sebelumnya, tekan tombol "Load Previous Results".\n""")

# --- User Input Section ---
query = st.text_input(
    "Kata Kunci:",
    placeholder='Masukkan kata kunci (Contoh: "Askrindo", "Waskita Karya")',
    key="search_query"
).lower()
 
colors = ["#0056a9"]
colorize_multiselect_options(colors)

# Sources
source_col1, source_col2 = st.columns([0.75, 0.3])
with source_col1:
    selected_sources = st.multiselect(
        "Website Sumber Berita",
        options=list(NEWS_SOURCES.keys()),
        default=list(NEWS_SOURCES.keys())
    )
with source_col2:
    max_items = st.text_input(
        "Jumlah Artikel Maksimal untuk tiap sumber",
        value="10",
        help="Data yang terambil dibatasi maksimal sekitar artikel per sumber untuk menghindari waktu tunggu yang lama."
    )

if max_items:
    if max_items.strip().isdigit():
        max_items = int(max_items)
    else:
        st.error("Error: Please enter a valid integer.")
        st.stop()

selected_news = [NEWS_SOURCES[source] for source in selected_sources]

# Date range inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Periode Awal:",
        key="start_date",
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        value = date(2020, 1, 1),
        help=("Catatan: tanggal awal dibatasi mulai 1 Januari 2020.")
    )
with col2:
    end_date = st.date_input(
        "Periode Akhir (opsional):",
        key="end_date",
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        value=date.today(),
        help=("Catatan: tanggal akhir dibatasi hingga hari ini dan tidak sebelum 1 Januari 2020.")
    )

if (start_date or end_date) and (start_date and end_date and start_date > end_date):
    st.error("Error: Start date must be before end date")
    st.stop()

# Additional sources
additional_sources = st.multiselect(
    "Sumber Informasi Tambahan:",
    options=list(ADDITIONAL_SOURCES.keys()),
    default=["INAPROC Daftar Hitam"],
    help = "Sumber informasi tambahan untuk pengecekan daftar hitam dan putusan MA"
)


if "inaproc_flag" not in st.session_state:
    st.session_state.inaproc_flag = False

if "putusan_ma_flag" not in st.session_state:
    st.session_state.putusan_ma_flag = False  # use consistent key name

if "user_click" not in st.session_state:
    st.session_state.user_click = 0

if "display_data" not in st.session_state:
    st.session_state.display_data = False

if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False

if "df_cached" not in st.session_state:
    st.session_state.df_cached = pd.DataFrame()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "user_feedback_count" not in st.session_state:
    st.session_state.user_feedback_count = 0

if st.session_state.df_cached.empty:
    MIN_FEEDBACK_REQUIRED = 0
else:
    MIN_FEEDBACK_REQUIRED = min(len(st.session_state.df_cached), 5)

# Now decide whether the button should be disabled
if not query.strip() or not selected_news:
    # invalid input → always disable button
    disable_button = True
else:
    if st.session_state.df_cached.empty:
        # no data collected yet → allow user to click
        disable_button = False
    else:
        # data already exists → disable until enough feedback given
        disable_button = st.session_state.user_feedback_count < MIN_FEEDBACK_REQUIRED

st.session_state.button_disabled = disable_button
stop_disabled = not st.session_state.df_cached.empty


col1, col2 = st.columns([2,4])   # Adjust ratio as you like

with col1:
    with stylable_container(
        "green_action_button",
        css_styles="""
            button {
                background-color: #a1d99b !important;
                color: black !important;
                border-radius: 6px !important;
                border: none !important;
            }

            button:disabled {
                background-color: #edf8e9 !important;
                color: #666666 !important;
                cursor: not-allowed !important;
            }
        """
    ):
        start_button = st.button(
            "Kumpulkan dan Analisa Data",
            disabled=st.session_state.button_disabled
        )

with col2:
    with stylable_container(
        "stop_button_red",
        css_styles="""
            button {
                background-color: #e74c3c !important;   /* red enabled */
                color: white !important;
                border-radius: 6px !important;
                border: none !important;
            }

            button:disabled {
                background-color: #f2b4ad !important;   /* light red / faded */
                color: #dddddd !important;               /* faded text */
                cursor: not-allowed !important;
            }
        """
    ):
        stop_button = st.button("Stop", disabled=stop_disabled)


st.markdown("<br>", unsafe_allow_html=True)   

df = pd.DataFrame()

if start_button:
    st.session_state.user_click += 1
    try:
        with st.spinner("Mengumpulkan artikel dan melakukan analisis", show_time=True):

            # --- main sentiment scraping ---
            df_isi = collect_data_and_analyze_sentiment(
                SENTIMENT_FILE_PATH,
                selected_news,
                max_items,
                query,
                start_date,
                end_date,
            )

            df = pd.concat([df, df_isi], ignore_index=True)
            df.to_csv(SENTIMENT_FILE_PATH, index=False)

            # --- additional sources: INAPROC ---
            if "INAPROC Daftar Hitam" in additional_sources:
                st.session_state.inaproc_flag = True
                inaproc_df = get_single_spider_data(
                    ADDITIONAL_SOURCES["INAPROC Daftar Hitam"],
                    INAPROC_FILE_PATH,
                    query,
                )
                inaproc_df.to_csv(INAPROC_FILE_PATH, index=False)
            else:
                st.session_state.inaproc_flag = False

            # --- additional sources: Putusan MA ---
            if "Putusan MA" in additional_sources:
                st.session_state.putusan_ma_flag = True
                putusan_ma_df = get_single_spider_data(
                    ADDITIONAL_SOURCES["Putusan MA"],
                    PUTUSAN_MA_FILE_PATH,
                    query,
                    start_date,
                    end_date
                )
                putusan_ma_df.to_csv(PUTUSAN_MA_FILE_PATH, index=False)
            else:
                st.session_state.putusan_ma_flag = False

            # --- clear caches ---
            load_data.clear()
            filter_data.clear()

            # --- reset feedback counter for THIS new result ---
            st.session_state.user_feedback_count = 0
            st.session_state.display_data = True

            st.session_state.df_cached = df

        st.success("Analisis selesai dan data telah disimpan.")
        st.rerun()   # only one rerun, after everything is done

    except Exception as e:
        st.error(f"An error occurred during scraping or analysis: {e}")
        st.exception(e)

    if st.session_state.last_query != query:
        st.session_state.df_cached = pd.DataFrame()
        st.session_state.user_feedback_count = 0
        st.session_state.last_query = query

if stop_button:
    st.stop()
    st.rerun()       

file_exists_and_not_empty = os.path.exists(SENTIMENT_FILE_PATH) and os.path.getsize(SENTIMENT_FILE_PATH) > 0
if st.button("Load Previous Results", disabled=not file_exists_and_not_empty):
    if file_exists_and_not_empty:
        st.info("Loading previously scraped data.")
        st.cache_data.clear()
        st.session_state.display_data = True
        # Reset feedback counter when loading previous results
        try:
            st.session_state['user_feedback_count'] = 0
        except Exception:
            pass
        if "INAPROC Daftar Hitam" in additional_sources:
            st.session_state.inaproc_flag = True
        else:
            st.session_state.inaproc_flag = False
        if "Putusan MA" in additional_sources:
            st.session_state.putusan_ma_flag = True
        else:
            st.session_state.putusan_ma_flag = False
        st.rerun()
    else:
        st.warning("No previous results found to load.")



if "display_data" not in st.session_state:
    st.session_state.display_data = False
st.sidebar.image(LOGO_PATH, use_container_width =True)

if st.session_state.display_data:
    df = load_data(SENTIMENT_FILE_PATH)
    if not df.empty:
        st.sidebar.subheader("Filter Data Hasil Analisis")
        filter_sources = None
        filter_sentiments = None
        filter_start_date = None
        filter_end_date = None

        if 'source' in df.columns:
            df['source'] = df['source'].replace({
                'cnn': 'CNN Indonesia',
                'kontan': 'Kontan',
                'kompas': 'Kompas',
                'cnbc': 'CNBC Indonesia'
            })
            all_sources = df['source'].unique()
            filter_sources = st.sidebar.multiselect(
                "Filter berdasarkan Sumber Berita:",
                options=all_sources,
                default=all_sources,
                key="global_source_filter"
            )

        if 'predicted_label' in df.columns:
            all_sentiments = df['predicted_label'].unique()
            filter_sentiments = st.sidebar.multiselect(
                "Filter Berdasarkan Hasil Sentimen:",
                options=all_sentiments,
                default=all_sentiments,
                key="global_sentiment_filter"
            )

        if 'date' in df.columns:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            filter_start_date, filter_end_date = st.sidebar.date_input(
                "Filter Rentang Waktu:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="global_date_filter"
            )

        df = filter_data(df, filter_sources, filter_sentiments, filter_start_date, filter_end_date)
        current_count = len(df)
        MIN_FEEDBACK_REQUIRED = min(current_count,5)
    
    st.sidebar.metric("Jumlah Progres Feedback", f"{st.session_state.user_feedback_count} / {MIN_FEEDBACK_REQUIRED}")            

    st.subheader("Hasil Analisis Sentimen")
    display_dataframe(df)
    if st.session_state.inaproc_flag:
        st.subheader("Hasil Pencarian Daftar Hitam INAPROC ")
        display_inaproc(query)

    if st.session_state.putusan_ma_flag:
        st.subheader("Hasil Pencarian Putusan Mahkamah Agung")
        display_putusan_ma(query)

    # Display pie chart and bar chart
    visualize_sentiment_distribution(df)
    # Display time series chart
    visualize_sentiment_over_time(df)
    # Display wordcloud
    visualize_wordcloud(df)
    # Display articles with sentiment
    display_articles(df)

# if stop_button:
#     st.stop()
#     st.rerun()

    streamlit_marquee(**{
    # the marquee container background color
    'background': "#d5d5d5",
    # the marquee text size
    'fontSize': '14px',
    # the marquee text color
    "color": "#000000",
    # the marquee text content
    'content': 'Mohon dapat mengisi feedback pada bagian artikel untuk membantu peningkatan kualitas model analisis sentimen. Terima kasih!',
    # the marquee container width
    'width': '1500px',
    # the marquee container line height
    'lineHeight': "14px",
    # the marquee duration
    'animationDuration': '20s'})

st.markdown("---")
