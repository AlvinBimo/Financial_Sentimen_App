from run_and_analyze import collect_data_and_analyze_sentiment, get_single_spider_data
from config import (TEMP_DIR, SENTIMENT_FILE_PATH, INAPROC_FILE_PATH, PUTUSAN_MA_FILE_PATH, DATA_DIR, FEEDBACK_FILE_PATH,
                    NEWS_SOURCES, ADDITIONAL_SOURCES, MODELS, SENTIMENT_COLORS)

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
from time import perf_counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from pathlib import Path


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
    pd.DataFrame(columns=['article_title', 'sentence', 'original_label', 'original_score', 'model',
                          'feedback', 'corrected_label', 'timestamp']).to_csv(FEEDBACK_FILE_PATH, index=False)


# --- Helper Functions ---
# Data Processing Functions
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(file_path: str, dropna: bool = True) -> pd.DataFrame:
    """
    Load data from CSV file into a pandas DataFrame.
    Returns empty DataFrame if file doesn't exist, is empty, or contains only headers.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 10:
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        if dropna:
            df = df.dropna()

        if df.empty:
            return pd.DataFrame()

        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(
                    df['date'].str.replace(r'\.\d+', '', regex=True),
                    errors='coerce'
                )
                df = df.dropna(subset=['date'])
                if df.empty:
                    return pd.DataFrame()
            except Exception as e:
                st.warning(f"Could not convert 'date' column to datetime: {e}")
                return pd.DataFrame()

        return df

    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def filter_data(
    df: pd.DataFrame,
    filter_sources: Optional[List[str]],
    filter_sentiments: Optional[List[str]],
    filter_start_date: Optional[date],
    filter_end_date: Optional[date]
) -> pd.DataFrame:
    """
    Applies source, sentiment, and date filter to loaded DataFrame.
    In this program, used to apply global filter before displays and visualizations.
    """
    if filter_sources and 'source' in df.columns:
        df = df[df['source'].isin(filter_sources)]

    if filter_sentiments and 'predicted_label' in df.columns:
        df = df[df['predicted_label'].isin(filter_sentiments)]

    if filter_start_date and filter_end_date and 'date' in df.columns:
        filter_start_date = pd.to_datetime(filter_start_date).normalize()
        filter_end_date = pd.to_datetime(filter_end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df = df[(df['date'] >= filter_start_date) & (df['date'] <= filter_end_date)]

    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_period_count(
        df: pd.DataFrame,
        time_period: Literal['Weekly', 'Monthly', 'Quarterly', 'Yearly']
) -> pd.DataFrame:
    """
    Groups DataFrame by periods to create data points for time-series visualization purposes.
    """
    df_display = df.copy()

    if time_period == "Weekly":
        df_display['period'] = df_display['date'].dt.to_period('W').dt.start_time
    elif time_period == "Monthly":
        df_display['period'] = df_display['date'].dt.to_period('M').dt.start_time
    elif time_period == "Quarterly":
        df_display['period'] = df_display['date'].dt.to_period('Q').dt.start_time
    else:
        df_display['period'] = df_display['date'].dt.to_period('Y').dt.start_time

    period_counts =  df_display.groupby(['period', 'predicted_label']).size().unstack(fill_value=0)

    for sentiment in SENTIMENT_COLORS:
        if sentiment not in period_counts.columns:
            period_counts[sentiment] = 0

    return period_counts[list(SENTIMENT_COLORS.keys())]


@st.cache_data(show_spinner=False, ttl=3600)
def get_sentiment_words(df: pd.DataFrame) -> dict:
    """
    Get all sentences in df for each sentiment, stem and remove stopwords, and combine for wordcloud visualization.
    """
    stemmer = StemmerFactory().create_stemmer()
    stop_words = StopWordRemoverFactory().get_stop_words()
    stop_words += ["pt", "persero", "group", "tbk"]
    dictionary = ArrayDictionary(stop_words)
    stopword_remover = StopWordRemover(dictionary)

    def lemmatize(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = stemmer.stem(text)
        text = stopword_remover.remove(text)
        return text

    sentiment_texts = {sentiment: '' for sentiment in SENTIMENT_COLORS}

    for idx, row in df.iterrows():
        title = row.get('title', '').lower()
        query = row.get('query', '').lower()

        # query is not guaranteed to be in title, but guaranteed to be in each sentence in sentences_pred
        if query in title.lower():
            title = title.replace(query, "")
            title_sentiment = ast.literal_eval(row.get('title_pred', [])).get('label', '')
            processed_title = lemmatize(title)
            if title_sentiment.strip() and processed_title.strip():
                sentiment_texts[title_sentiment] += processed_title

        sentences_pred = row.get('sentences_pred', [])

        if isinstance(sentences_pred, str):
            try:
                sentences_pred = re.sub(r'}\s*{', '}, {', sentences_pred)
                sentences_pred = ast.literal_eval(sentences_pred)
            except (ValueError, SyntaxError):
                sentences_pred = []

        if sentences_pred and isinstance(sentences_pred, list):
            for sentence_data in sentences_pred:
                if isinstance(sentence_data, dict):
                    sentence_sentiment = sentence_data.get('label', '').lower()
                    sentence_text = sentence_data.get('sentence', '').lower()
                    sentence_text = sentence_text.replace(query, "")

                    if sentence_sentiment in sentiment_texts and sentence_text:
                        processed_sentence = lemmatize(sentence_text)
                        sentiment_texts[sentence_sentiment] += ' ' + processed_sentence

    return sentiment_texts


def save_feedback(feedback_data: Dict) -> None:
    """
    Saves model prediction feedback data from user.
    """
    try:
        df = pd.read_csv(FEEDBACK_FILE_PATH)
        new_df = pd.DataFrame([feedback_data])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(FEEDBACK_FILE_PATH, index=False)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")


# Display/Visualization Functions
def display_dataframe(
        df: pd.DataFrame,
        filename: str = "data"
) -> None:
    """
    Displays subheader section displaying a DataFrame with download buttons.
    """
    if df.empty:
        return

    st.dataframe(df, use_container_width=True)

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f'{filename}.csv',
            mime='text/csv',
            key=f'dl_{filename}_csv'
        )
    with dl_col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f'{filename}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key=f'dl_{filename}_xlsx'
        )


def visualize_sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Displays subheader section visualizing sentiment distributions in a DataFrame.
    Plots: Pie chart (Overall distribution), Bar chart (per source)
    """
    st.subheader("Distribusi Sentimen Keseluruhan")

    if df.empty or 'predicted_label' not in df.columns:
        st.info("Sentiment analysis results kosong untuk ditampilkan.")
        return

    fig_pie = px.pie(
        df,
        names='predicted_label',
        title='Overall Sentiment Distribution',
        color='predicted_label',
        color_discrete_map=SENTIMENT_COLORS,
        hover_data=['predicted_label'],
        labels={'count': 'Count'}
    )
    fig_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Sentiment Distribution by News Source")

    view_type = st.radio(
        "Display sentiment as:",
        ["Count", "Percentage"],
        horizontal=True,
        key="sentiment_distribution_view"
    )

    source_sentiment = pd.crosstab(df['source'], df['predicted_label'])

    for sentiment in SENTIMENT_COLORS:
        if sentiment not in source_sentiment.columns:
            source_sentiment[sentiment] = 0

    source_sentiment = source_sentiment[list(SENTIMENT_COLORS.keys())]

    fig_stacked = go.Figure()

    if view_type == "Percentage":
        data = source_sentiment.div(source_sentiment.sum(axis=1), axis=0) * 100
        yaxis_title = "Percentage"
        hover_template = "<b>%{fullData.name}</b>: %{y:.1f}%<extra></extra>"
        yaxis_range = [0, 100]
    else:
        data = source_sentiment
        yaxis_title = "Number of Articles"
        hover_template = "<b>%{fullData.name}</b>: %{y}<extra></extra>"
        yaxis_range = None

    for sentiment in SENTIMENT_COLORS:
        fig_stacked.add_trace(go.Bar(
            x=data.index,
            y=data[sentiment],
            name=sentiment.capitalize(),
            marker_color=SENTIMENT_COLORS[sentiment],
            hovertemplate=hover_template
        ))

    fig_stacked.update_layout(
        barmode='stack',
        title=f'Distribusi Sentimen dari Tiap Sumber Berita({view_type})',
        xaxis_title='News Source',
        yaxis_title=yaxis_title,
        hovermode='x unified',
        height=500,
        yaxis=dict(range=yaxis_range)
    )

    st.plotly_chart(fig_stacked, use_container_width=True)


def visualize_sentiment_over_time(df: pd.DataFrame) -> None:
    """
    Displays subheader section with a time-series visualization of sentiments in a DataFrame.
    Plots: Area plot
    """
    st.subheader("Sentiment Over Time")

    if df.empty or 'predicted_label' not in df.columns or 'date' not in df.columns:
        st.info("Sentiment analysis results kosong untuk ditampilkan.")
        return

    time_period = st.radio(
        "Time Period:",
        ["Weekly", "Monthly", "Quarterly", "Yearly"],
        horizontal=True,
        key="time_period_radio_widget"
    )

    display_option = st.radio(
        "Display sentiment as:",
        ["Count", "Percentage"],
        horizontal=True,
        key="sentiment_display_option_radio_widget",
    )

    period_counts = get_period_count(df, time_period)

    fig_time = go.Figure()

    if display_option == "Count":
        for sentiment in SENTIMENT_COLORS:
            fig_time.add_trace(go.Scatter(
                x=period_counts.index.astype(str),
                y=period_counts[sentiment],
                name=sentiment.capitalize(),
                line=dict(color=SENTIMENT_COLORS[sentiment], width=2),
                mode='lines+markers',
                marker=dict(size=8),
                stackgroup='one'
            ))
        yaxis_title = "Number of Articles"
    else:
        period_totals = period_counts.sum(axis=1)
        period_ratios = period_counts.div(period_totals.replace(0, 1), axis=0).fillna(0)

        for sentiment in SENTIMENT_COLORS:
            fig_time.add_trace(go.Scatter(
                x=period_ratios.index.astype(str),
                y=period_ratios[sentiment],
                name=sentiment.capitalize(),
                line=dict(color=SENTIMENT_COLORS[sentiment], width=2),
                mode='lines+markers',
                marker=dict(size=8),
                stackgroup='one'
            ))
        yaxis_title = "Sentiment Ratio"
        fig_time.update_layout(yaxis=dict(tickformat=".0%"))

    fig_time.update_layout(
        title=f'Sentiment Trend Over Time ({time_period})',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_time, use_container_width=True)


def visualize_wordcloud(df: pd.DataFrame) -> None:
    """
    Create wordclouds for each sentiment category
    """
    st.subheader("Sentiment Wordcloud")

    if df.empty or 'title' not in df.columns or 'predicted_label' not in df.columns:
        st.info("Wordcloud kosong karena tidak ada artikel yang ditampilan")
        return

    sentiment_texts = get_sentiment_words(df)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for i, (sentiment, text) in enumerate(sentiment_texts.items()):
        if text.strip():
            wordcloud = WordCloud(
                background_color='white',
                colormap='viridis' if sentiment == 'positive' else 'cool' if sentiment == 'neutral' else 'Reds',
                max_words=30,
                contour_width=3,
                contour_color='steelblue',
                random_state=15
            ).generate(text)

            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment.capitalize()} Sentiment', fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No {sentiment} sentiment data',
                         ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{sentiment.capitalize()} Sentiment', fontsize=14, fontweight='bold')
            axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


def display_articles(df: pd.DataFrame) -> None:
    """
    Displays articles in a DataFrame with:
     - Pagination
     - Checkbox to highlight most relevant sentence in an article
     - Dropdown to show all relevant sentence and model predictions in each article
     - User rating feature for model predictions
    """
    st.subheader("Hasil Analisis Sentimen berserta Artikel")

    if df.empty or 'title' not in df.columns:
        st.info("Artikel kosong untuk ditampilkan.")
        return

    df_display = df.copy()

    if 'predicted_label' not in df_display.columns:
        df_display['predicted_label'] = 'unknown'
    else:
        df_display['predicted_label'] = df_display['predicted_label'].fillna('unknown')

    sort_option = st.selectbox(
        "Sort by:",
        options=["Date (Newest First)", "Date (Oldest First)", "Sentiment"],
        key="sort_option"
    )

    if sort_option == "Date (Newest First)":
        if 'date' in df_display.columns:
            df_display = df_display.sort_values('date', ascending=False)
    elif sort_option == "Date (Oldest First)":
        if 'date' in df_display.columns:
            df_display = df_display.sort_values('date', ascending=True)
    elif sort_option == "Sentiment":
        if 'predicted_label' in df_display.columns:
            df_display = df_display.sort_values('predicted_label')

    article_options_col1, article_options_col2 = st.columns(2)

    with article_options_col1:
        items_per_page = st.radio(
            "Items per page:",
            [10, 50, 100],
            horizontal=True,
            key="items_per_page_radio_widget"
        )
    with article_options_col2:
        highlight_relevant_sentence = st.checkbox("Highlight Relevant Sentence")

    total_pages = (len(df_display) // items_per_page) + (1 if len(df_display) % items_per_page else 0)

    if total_pages > 1:
        page = st.number_input(
            "Page:",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="page_number"
        )
    else:
        page = 1

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_df = df_display.iloc[start_idx:end_idx]

    for idx, row in paginated_df.iterrows():
        sentiment = row.get('predicted_label', 'unknown')
        color = SENTIMENT_COLORS.get(sentiment, '#9E9E9E')

        full_body = row.get('body', 'No content')
        sentences_pred = row.get('sentences_pred', [])
        # literal eval when we need to use it instead of when we're loading
        if isinstance(sentences_pred, str):
            try:
                # Note: HF datasets saves python dictionary in list objects in csv differently than pandas dataframe
                # Using pandas won't need the workaround below as the commas are included
                sentences_pred = re.sub(r'}\s*{', '}, {', sentences_pred)
                sentences_pred = ast.literal_eval(sentences_pred)
            except:
                sentences_pred = []

        display_body = full_body[:300] + ('...' if len(full_body) > 300 else '')

        if highlight_relevant_sentence and sentences_pred and isinstance(sentences_pred, list):
            same_sentiment_sentences = [s for s in sentences_pred
                                        if isinstance(s, dict) and
                                        s.get('label', '').lower() == sentiment.lower()]

            if same_sentiment_sentences:
                most_relevant = max(same_sentiment_sentences, key=lambda x: x.get('score', 0))
                sentence_to_highlight = most_relevant.get('sentence', '')

                if sentence_to_highlight and sentence_to_highlight in full_body:
                    sentence_start = full_body.find(sentence_to_highlight)
                    if sentence_start >= 0:
                        sentence_end = sentence_start + len(sentence_to_highlight)

                        context_start = max(0, sentence_start - 150)
                        context_end = min(len(full_body), sentence_end + 150)

                        context = full_body[context_start:context_end]

                        highlighted_sentence = f"""
                        <span style="background-color: {color}60; font-weight: bold;">
                            {sentence_to_highlight}
                        </span>
                        """
                        context = context.replace(sentence_to_highlight, highlighted_sentence)

                        prefix = '...' if context_start > 0 else ''
                        suffix = '...' if context_end < len(full_body) else ''

                        display_body = f"{prefix}{context}{suffix}"

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: {color}20;
                    border-left: 5px solid {color};
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                ">
                    <a href="{row.get('link', '#')}"><h4>{row.get('title', 'No title')}</h4></a>
                    <p><strong>Source:</strong> {row.get('source', 'Unknown')} | 
                    <strong>Date:</strong> {row.get('date', 'Unknown')} | 
                    <strong>Sentiment:</strong> {sentiment.capitalize()}</p>
                    <p>{display_body}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # TODO: edge case when no relevant sentence in article body. overall sentiment will be determined by
            #       title, but user cannot see and rate the prediction in the UI.
            if 'sentences_pred' in row and row['sentences_pred']:
                with st.expander("Show relevant sentences"):
                    if isinstance(sentences_pred, list) and sentences_pred:
                        st.write("Relevant sentences with their sentiment scores:")
                        for i, sent in enumerate(sentences_pred):
                            if isinstance(sent, dict):
                                sent_color = SENTIMENT_COLORS.get(sent.get('label', 'unknown').lower(), '#9E9E9E')

                                col1, col2 = st.columns([0.9, 0.1])

                                with col1:
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: {sent_color}20;
                                            border-left: 3px solid {sent_color};
                                            padding: 8px;
                                            margin: 5px 0;
                                            border-radius: 3px;
                                        ">
                                            <p><strong>Label:</strong> {sent.get('label', 'unknown').capitalize()} | 
                                            <strong>Score:</strong> {sent.get('score', 0):.2f}</p>
                                            <p>{sent.get('sentence', '')}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                with col2:
                                    feedback_key = f"feedback_{idx}_{i}"
                                    good_review = st.button("👍", key=f"up_{feedback_key}")
                                    bad_review = st.button("👎", key=f"down_{feedback_key}")

                                if f'open_label_correction_{feedback_key}' not in st.session_state:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = False

                                if good_review:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = False
                                    feedback_data = {
                                        'article_title': row.get('title', ''),
                                        'sentence': sent.get('sentence', ''),
                                        'original_label': sent.get('label', ''),
                                        'original_score': sent.get('score', 0),
                                        'model': st.session_state.model_path,
                                        'feedback': 'good',
                                        'corrected_label': '',
                                        'timestamp': pd.Timestamp.now()
                                    }
                                    save_feedback(feedback_data)
                                    st.success("Thanks for your feedback!")

                                if bad_review:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = True

                                if st.session_state[f'open_label_correction_{feedback_key}']:
                                    with st.form(key=f"correction_form_{feedback_key}"):
                                        corrected_label = st.selectbox(
                                            "What should the correct sentiment be?",
                                            options=[l for l in ["positive", "negative", "neutral"] if
                                                     l != sent.get('label', '')],
                                            key=f"corrected_label_{feedback_key}"
                                        )
                                        submitted = st.form_submit_button("Submit Correction")

                                        if submitted:
                                            feedback_data = {
                                                'article_title': row.get('title', ''),
                                                'sentence': sent.get('sentence', ''),
                                                'original_label': sent.get('label', ''),
                                                'original_score': sent.get('score', 0),
                                                'model': st.session_state.model_path,
                                                'feedback': 'bad',
                                                'corrected_label': corrected_label,
                                                'timestamp': pd.Timestamp.now()
                                            }
                                            save_feedback(feedback_data)
                                            st.success("Thanks for your feedback!")
                    else:
                        st.info("No relevant sentences available")

    if total_pages > 1:
        st.write(
            f"Showing articles {start_idx + 1}-{min(end_idx, len(df_display))} of {len(df_display)} (Page {page}/{total_pages})")
    else:
        st.write(f"Showing {len(df_display)} articles")


def display_inaproc() -> None:
    """
    Checks if Daftar Hitam INAPROC data was gathered and displays it if exists.
    """
    # measure load time for inaproc data
    start = perf_counter()
    inaproc_df = load_data(INAPROC_FILE_PATH, False)
    end = perf_counter()
    inaproc_sec = max(0.0, end - start)
    try:
        st.session_state['inaproc_seconds'] = inaproc_sec
    except Exception:
        pass

    if not inaproc_df.empty:
        st.error(f"{query.title()} Ditemukan pada Daftar Hitam INAPROC")
        display_dataframe(inaproc_df, "inaproc")
    else:
        st.success(f"{query.title()} Tidak Ditemukan di Daftar Hitam INAPROC")


def display_putusan_ma() -> None:
    """
    Checks if Putusan MA data was gathered and displays it if exists.
    """
    # measure load time for putusan_ma data
    start = perf_counter()
    putusan_ma_df = load_data(PUTUSAN_MA_FILE_PATH, False)
    end = perf_counter()
    putusan_ma_sec = max(0.0, end - start)
    try:
        st.session_state['putusan_ma_seconds'] = putusan_ma_sec
    except Exception:
        pass

    if not putusan_ma_df.empty:
        st.error(f"{query.title()} Ditemukan di Putusan Mahkamah Agung")
        display_dataframe(putusan_ma_df, "putusan_ma")
    else:
        st.success(f"{query.title()} Tidak Ditemukan di Putusan Mahkamah Agung")


# --- Streamlit App Layout ---
# language = st.sidebar.selectbox(
#     label='Language',
#     options=['English', 'Indonesian'],
#     key="language"
# )

st.title("📊 Aplikasi Analisis Reputasi Calon Terjamin")

st.write("""
Masukkan kata kunci pencarian Anda dan pilih sumber untuk mengumpulkan data dan melakukan analisis sentimen.\n
Untuk melihat data yang telah dikumpulkan sebelumnya, tekan tombol "Load Previous Results".\n""")

query = st.text_input(
    "Kata Kunci:",
    placeholder='Masukkan kata kunci (Contoh: "Askrindo", "Waskita Karya")',
    key="search_query"
).lower()

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
        placeholder="100",
        value="100",
        help="Data yang diambil tidak selalu tepat sejumlah ini (mungkin sedikit lebih banyak)."
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
        "Periode Awal (Opsional):",
        value=None,
        key="start_date",
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        help=("Pemilihan periode awal dan akhir bersifat opsional. Jika tidak diisi, "
              "maka artikel yang akan diambil adalah artikel dari seluruh periode. "
              "Catatan: tanggal awal dibatasi mulai 1 Januari 2020.")
    )
with col2:
    end_date = st.date_input(
        "Periode Akhir (opsional):",
        value=None,
        key="end_date",
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        help=("Pemilihan periode awal dan akhir bersifat opsional. Jika tidak diisi, "
              "maka artikel yang akan diambil adalah artikel dari seluruh periode."
              "Catatan: tanggal akhir dibatasi hingga hari ini dan tidak sebelum 1 Januari 2020.")
    )

if (start_date or end_date) and (start_date and end_date and start_date > end_date):
    st.error("Error: Start date must be before end date")
    st.stop()

# Model selection
model = st.selectbox(
    "Model:",
    options=list(MODELS.keys()),
    help="Choose which model to use for sentiment analysis",
    key="model"
)

# Additional sources
additional_sources = st.multiselect(
    "Sumber Informasi Tambahan:",
    options=list(ADDITIONAL_SOURCES.keys()),
    default=list(ADDITIONAL_SOURCES.keys()),
    help = "Sumber informasi tambahan untuk pengecekan daftar hitam dan putusan MA"
)

# --- Main Sentiment Analysis Functionality ---
if "model_path" not in st.session_state:
    st.session_state.model_path = ''

if "inaproc_flag" not in st.session_state:
    st.session_state.inaproc_flag = False

if "putusan_ma_flag" not in st.session_state:
    st.session_state.putusan_ma = False

if st.button("Kumpulkan dan Analisa Data", disabled=(not query.strip() or not selected_news)):
    st.session_state.display_data = False

    if not query.strip():
        st.warning("Please enter a valid search query")
        st.stop()

    if not selected_news:
        st.warning("Please select at least one news source")
        st.stop()

    model_path = MODELS.get(model, None)

    if not model_path:
        st.warning("Please select a valid model")
        st.stop()

    if start_date and not end_date:
        st.warning("Please specify an end date")
        st.stop()
    elif end_date and not start_date:
        st.warning("Please specify a start date")
        st.stop()

    try:
        with st.spinner("Mengumpulkan artikel dan melakukan analisis", show_time=True):
            df = collect_data_and_analyze_sentiment(SENTIMENT_FILE_PATH, selected_news, max_items, query, start_date,
                                                    end_date, model_path)

            df.to_csv(SENTIMENT_FILE_PATH, index=False)

            if "INAPROC Daftar Hitam" in additional_sources:
                st.session_state.inaproc_flag = True
                inaproc_df = get_single_spider_data(ADDITIONAL_SOURCES["INAPROC Daftar Hitam"], INAPROC_FILE_PATH, query)
                inaproc_df.to_csv(INAPROC_FILE_PATH, index=False)
            else:
                st.session_state.inaproc_flag = False

            if "Putusan MA" in additional_sources:
                st.session_state.putusan_ma_flag = True
                putusan_ma_df = get_single_spider_data(ADDITIONAL_SOURCES["Putusan MA"], PUTUSAN_MA_FILE_PATH, query)
                putusan_ma_df.to_csv(PUTUSAN_MA_FILE_PATH, index=False)
            else:
                st.session_state.putusan_ma_flag = False

            st.session_state.model_path = model_path

        st.success("Analisis selesai dan data telah disimpan.")

        load_data.clear()
        filter_data.clear()

        st.session_state.display_data = True
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred during scraping or analysis: {e}")
        st.exception(e)

file_exists_and_not_empty = os.path.exists(SENTIMENT_FILE_PATH) and os.path.getsize(SENTIMENT_FILE_PATH) > 0
if st.button("Load Previous Results", disabled=not file_exists_and_not_empty):
    if file_exists_and_not_empty:
        st.info("Loading previously scraped data.")
        st.cache_data.clear()
        st.session_state.display_data = True
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

if st.session_state.display_data:
    df = load_data(SENTIMENT_FILE_PATH)
    # If timings were recorded during the last run, display them as metrics
    timings = st.session_state.get('timings', None)
    if timings:
        try:
            scrape_sec = timings.get('scrape_seconds', 0.0)
            infer_sec = timings.get('inference_seconds', 0.0)
            total_sec = timings.get('total_seconds', scrape_sec + infer_sec)
            Inaproc_sec = st.session_state.get('inaproc_seconds', None)
            putusan_ma_sec = st.session_state.get('putusan_ma_seconds', None)

            tcol1, tcol2, tcol3= st.columns(3)
            tcol1.metric(label="Lama Waktu Pencarian", value=f"{scrape_sec:.2f}s")
            tcol2.metric(label="Lama Waktu Analisa", value=f"{infer_sec:.2f}s" if infer_sec==0.0 else "N/A")
            tcol3.metric(label="Total Waktu Proses", value=f"{total_sec:.2f}s")

            _, tcols4, tcols5, _ = st.columns([0.3, 1, 1, 0.1])
            tcols4.metric(label="Waktu Cek INAPROC", value=f"{Inaproc_sec:.2f}s" if Inaproc_sec==0.0 else "N/A")
            tcols5.metric(label="Waktu Cek Putusan", value=f"{putusan_ma_sec:.2f}s" if putusan_ma_sec==0.0 else "N/A")
        except Exception:   
            pass
            # show additional timings for INAPROC and Putusan MA if available
            
        #     if inaproc_sec is not None or putusan_ma_sec is not None:
        #         cols = st.columns(2)
        #         if inaproc_sec is not None:
        #             cols[0].metric(label="Waktu Cek INAPROC", value=f"{inaproc_sec:.2f}s")
        #         else:
        #             cols[0].write("")
        #         if putusan_ma_sec is not None:
        #             cols[1].metric(label="Waktu Cek Putusan MA", value=f"{putusan_ma_sec:.2f}s")
        #         else:
        #             cols[1].write("")
        # except Exception:
        #     # If anything goes wrong while reading/displaying timings, ignore silently

    if not df.empty:
        st.sidebar.subheader("Global Filters")
        filter_sources = None
        filter_sentiments = None
        filter_start_date = None
        filter_end_date = None

        if 'source' in df.columns:
            all_sources = df['source'].unique()
            filter_sources = st.sidebar.multiselect(
                "Filter by Source:",
                options=all_sources,
                default=all_sources,
                key="global_source_filter"
            )

        if 'predicted_label' in df.columns:
            all_sentiments = df['predicted_label'].unique()
            filter_sentiments = st.sidebar.multiselect(
                "Filter by Sentiment:",
                options=all_sentiments,
                default=all_sentiments,
                key="global_sentiment_filter"
            )

        if 'date' in df.columns:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            filter_start_date, filter_end_date = st.sidebar.date_input(
                "Filter by Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="global_date_filter"
            )

        df = filter_data(df, filter_sources, filter_sentiments, filter_start_date, filter_end_date)

    if st.session_state.inaproc_flag:
        st.subheader("Hasil Pencarian Daftar Hitam INAPROC ")
        display_inaproc()

    if st.session_state.putusan_ma_flag:
        st.subheader("Hasil Pencarian Putusan Mahkamah Agung")
        display_putusan_ma()

    # Display pie chart and bar chart
    visualize_sentiment_distribution(df)
    # Display time series chart
    visualize_sentiment_over_time(df)
    # Display wordcloud
    visualize_wordcloud(df)
    # Display articles with sentiment
    display_articles(df)
st.markdown("---")
