from run_and_analyze import collect_data_and_analyze_sentiment, get_single_spider_data
from config import (TEMP_DIR, SENTIMENT_FILE_PATH, INAPROC_FILE_PATH, PUTUSAN_MA_FILE_PATH, DATA_DIR, FEEDBACK_FILE_PATH,
                    NEWS_SOURCES, ADDITIONAL_SOURCES,SENTIMENT_COLORS, LOGO_PATH)

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
    stop_words += ["pt", "persero", "group", "tbk","rp",'nggak','nya','nih','si','deh','kok','gue','loh','loh','yah','tau','aja']
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
    Saves or updates model prediction feedback data from user.
    - If feedback for the same article+sentence already exists, replace it and
      DO NOT increase user_feedback_count.
    - Otherwise, append new feedback and increase the counter.
    """

    feedback_id = feedback_data.get("feedback_id")

    # Load existing feedback file if exists, otherwise create empty DataFrame
    try:
        df = pd.read_csv(FEEDBACK_FILE_PATH)
    except Exception:
        df = pd.DataFrame()

    # Ensure feedback_id column exists
    if "feedback_id" not in df.columns:
        if df.empty:
            df = pd.DataFrame(columns=["feedback_id"])
        else:
            df["feedback_id"] = None

    # Check if this feedback_id already exists
    existing_idx = []
    if not df.empty:
        existing_idx = df.index[df["feedback_id"] == feedback_id].tolist()

    if existing_idx:
        # Replace existing feedback row
        df.loc[existing_idx[0]] = feedback_data
        is_new_feedback = False
    else:
        # Append as new feedback
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
        is_new_feedback = True

    # Save back to CSV
    try:
        df.to_csv(FEEDBACK_FILE_PATH, index=False)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return

    # Only increment counter for NEW feedback
    if "user_feedback_count" not in st.session_state:
        st.session_state.user_feedback_count = 0

    if is_new_feedback:
        st.session_state.user_feedback_count = min(
            st.session_state.user_feedback_count + 1,
            20
        )


# Display/Visualization Functions
def display_dataframe(
        df: pd.DataFrame,
        filename: str = "data"
) -> None:
    """
    Displays subheader section displaying a DataFrame with download buttons.
    """
    # st.subheader("Tabel Hasil Pengumpulan Data")
    if df.empty:
        return

    st.dataframe(df, 
                 use_container_width=True, 
                 hide_index=True,
                 column_config= {"link" : st.column_config.LinkColumn()})

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f'{filename}_{query}.csv',
            mime='text/csv',
            key=f'dl_{filename}_csv'
        )
    with dl_col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(col)
                )
                worksheet.set_column(i, i, max_len)
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f'{filename}_{query}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key=f'dl_{filename}_xlsx'
        )

def visualize_sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Displays subheader section visualizing sentiment distributions in a DataFrame.
    Plots: Pie chart (Overall distribution), Bar chart (per source)
    """
    with st.container(border = True) :
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
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",                
            insidetextfont=dict(size=16, color='white'),
            outsidetextfont=dict(size=14, color='black'),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with st.container(border = True) :
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

        st.plotly_chart(
            fig_stacked,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d"],"scrollZoom": False}
        )

def visualize_sentiment_over_time(df: pd.DataFrame) -> None:
    """
    Displays subheader section with a time-series visualization of sentiments in a DataFrame.
    Plots: Area plot
    """
    with st.container(border = True) :
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
        st.plotly_chart(
            fig_time,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d"],"scrollZoom": False}
        )

def visualize_wordcloud(df: pd.DataFrame) -> None:
    """
    Create wordclouds for each sentiment category
    """
    st.subheader("Sentiment Wordcloud")

    if df.empty or 'title' not in df.columns or 'predicted_label' not in df.columns:
        st.info("Wordcloud kosong karena tidak ada artikel yang ditampilan")
        return

    sentiment_texts = get_sentiment_words(df)

    fig, axes = plt.subplots(1, 3, figsize=(12, 50))

    for i, (sentiment, text) in enumerate(sentiment_texts.items()):
        if text.strip():
            wordcloud = WordCloud(
                background_color='white',
                colormap='Greens' if sentiment == 'positive' else 'cool' if sentiment == 'neutral' else 'Reds',
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
    st.pyplot(fig, use_container_width=True)

st.session_state.user_feedback_count = st.session_state.get('user_feedback_count', 0)
MIN_FEEDBACK_REQUIRED = 5        

def display_articles(df: pd.DataFrame) -> None:

    # Display progress in sidebar
    st.sidebar.markdown("---")
    df_display = df.copy()

    st.subheader("Hasil Analisis Sentimen berserta Artikel")
    st.warning(
        f"Mohon berikan feedback terhadap hasil analisis sentimen pada artikel di bawah ini.\n"
        f"Saat ini Anda telah memberikan {st.session_state.user_feedback_count} dari "
        f"{MIN_FEEDBACK_REQUIRED} feedback yang diperlukan."
    )

    if df_display.empty:
        st.info("Artikel kosong untuk ditampilkan.")
        return

    # Sorting feature
    sort_option = st.selectbox(
        "Urut berdasarkan:",
        options=["Tanggal (Paling Baru)", "Tanggal (Paling Lama)", "Hasil Sentimen"],
        key="sort_option"
    )

    if sort_option == "Tanggal (Paling Baru)":
        df_display = df_display.sort_values('date', ascending=False)
    elif sort_option == "Tanggal (Paling Lama)":
        df_display = df_display.sort_values('date', ascending=True)
    elif sort_option == "Hasil Sentimen":
        df_display = df_display.sort_values('predicted_label')

    article_options_col1, article_options_col2 = st.columns(2)

    with article_options_col1:
        items_per_page = st.radio(
            "Jumlah Artikel Per Halaman:",
            [10, 50, 100],
            horizontal=True,
            key="items_per_page_radio_widget"
        )
    with article_options_col2:
        highlight_relevant_sentence = st.checkbox(
            "Highlight Kalimat yang Paling Relevan",
            value=True
        )

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
        if isinstance(sentences_pred, str) and sentences_pred:
            try:
                sentences_pred = re.sub(r'}\s*{', '}, {', sentences_pred)
                sentences_pred = ast.literal_eval(sentences_pred)
            except Exception:
                sentences_pred = []

        display_body = full_body[:300] + ('...' if len(full_body) > 300 else '')

        if highlight_relevant_sentence and sentences_pred and isinstance(sentences_pred, list):
            same_sentiment_sentences = [
                s for s in sentences_pred
                if isinstance(s, dict)
                and s.get('label', '').lower() == str(sentiment).lower()
            ]

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
                    <strong>Sentiment:</strong> {str(sentiment).capitalize()}</p>
                    <p>{display_body}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if 'sentences_pred' in row and row['sentences_pred']:
                with st.expander("Show relevant sentences"):
                    unique_sentences = []
                    seen = set()

                    for sent in sentences_pred:
                        if not isinstance(sent, dict):
                            continue
                        key = (sent.get("sentence", ""), sent.get("label", ""))
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_sentences.append(sent)
                    if isinstance(unique_sentences, list) and sentences_pred:
                        st.write("Kalimat Berkaitan dan Prediksi Sentimennya:")
                        for i, sent in enumerate(unique_sentences):
                            if isinstance(sent, dict):
                                sent_label = sent.get('label', 'unknown').lower()
                                sent_color = SENTIMENT_COLORS.get(sent_label, '#9E9E9E')

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
                                            <p><strong>Hasil Sentimen:</strong> {sent.get('label', 'unknown').capitalize()} | 
                                            <strong>Skor:</strong> {sent.get('score', 0):.2f}</p>
                                            <p>{sent.get('sentence', '')}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                # Unique key for this feedback item
                                feedback_key = f"feedback_{idx}_{i}"
                                feedback_id = f"{row.get('title', '')}::{sent.get('sentence', '')}"

                                with col2:
                                    good_review = st.button("👍", key=f"up_{feedback_key}")
                                    bad_review = st.button("👎", key=f"down_{feedback_key}")

                                if f'open_label_correction_{feedback_key}' not in st.session_state:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = False

                                if good_review:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = False
                                    feedback_data = {
                                        'feedback_id': feedback_id,
                                        'article_title': row.get('title', ''),
                                        'sentence': sent.get('sentence', ''),
                                        'original_label': sent.get('label', ''),
                                        'original_score': sent.get('score', 0),
                                        'feedback': 'good',
                                        'corrected_label': '',
                                        'timestamp': pd.Timestamp.now()
                                    }
                                    save_feedback(feedback_data)
                                    st.sidebar.metric(
                                        "Jumlah Progres Feedback",
                                        f"{st.session_state.user_feedback_count} / {MIN_FEEDBACK_REQUIRED}"
                                    )
                                    st.rerun()

                                if bad_review:
                                    st.session_state[f'open_label_correction_{feedback_key}'] = True

                                if st.session_state[f'open_label_correction_{feedback_key}']:
                                    with st.form(key=f"correction_form_{feedback_key}"):
                                        corrected_label = st.selectbox(
                                            "What should the correct sentiment be?",
                                            options=[
                                                l for l in ["positive", "negative", "neutral"]
                                                if l != sent.get('label', '')
                                            ],
                                            key=f"corrected_label_{feedback_key}"
                                        )
                                        submitted = st.form_submit_button("Submit Correction")

                                        if submitted:
                                            feedback_data = {
                                                'feedback_id': feedback_id,
                                                'article_title': row.get('title', ''),
                                                'sentence': sent.get('sentence', ''),
                                                'original_label': sent.get('label', ''),
                                                'original_score': sent.get('score', 0),
                                                'feedback': 'bad',
                                                'corrected_label': corrected_label,
                                                'timestamp': pd.Timestamp.now()
                                            }
                                            save_feedback(feedback_data)
                                            st.session_state[f'open_label_correction_{feedback_key}'] = False
                                            st.sidebar.metric(
                                                "Jumlah Progres Feedback",
                                                f"{st.session_state.user_feedback_count} / {MIN_FEEDBACK_REQUIRED}"
                                            )
                                            st.rerun()
                    else:
                        st.info("No relevant sentences available")

    if total_pages > 1:
        st.write(
            f"Showing articles {start_idx + 1}-{min(end_idx, len(df_display))} "
            f"of {len(df_display)} (Page {page}/{total_pages})"
        )
    else:
        st.write(f"Showing {len(df_display)} articles")

def display_inaproc() -> None:
    """
    Checks if Daftar Hitam INAPROC data was gathered and displays it if exists.
    """
    # measure load time for inaproc data
    inaproc_df = load_data(INAPROC_FILE_PATH, False)
    if not inaproc_df.empty:
        st.error(f"{query.title()} **Ditemukan** pada Daftar Hitam INAPROC")
        display_dataframe(inaproc_df, "inaproc")
    else:
        st.success(f"{query.title()} **Tidak Ditemukan** di Daftar Hitam INAPROC")


def display_putusan_ma() -> None:
    """
    Checks if Putusan MA data was gathered and displays it if exists.
    """
    # measure load time for putusan_ma data
    st.write("Data Putusan Mahkamah Agung diambil dari website resmi Mahkamah Agung Republik Indonesia.")
    st.write("Hasil pencarian Putusan Mahkamah Agung mungkin tidak lengkap karena keterbatasan teknis pada website resmi Mahkamah Agung Republik Indonesia.")
    putusan_ma_df = load_data(PUTUSAN_MA_FILE_PATH, False)
    if not putusan_ma_df.empty:
        st.error(f"{query.title()} **Ditemukan** di Putusan Mahkamah Agung")
        display_dataframe(putusan_ma_df, "putusan_ma")
    else:
        st.success(f"{query.title()} **Tidak Ditemukan** di Putusan Mahkamah Agung")


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

def colorize_multiselect_options(colors: list[str]) -> None:
    rules = ""
    n_colors = len(colors)
 
    for i, color in enumerate(colors):
        rules += f"""
        .stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({n_colors}n+{i}) {{
            background-color: {color} !important;
            color: white !important;
        }}
        .stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({n_colors}n+{i}) svg {{
            fill: white !important;
        }}
        """
 
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)
 
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

# if query != st.session_state.last_query:
#     st.session_state.last_query = query
#     st.session_state.df_cached = pd.DataFrame()
#     st.session_state.user_feedback_count = 0
#     st.session_state.button_disabled = True

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
                background-color: #e74c3c !important;   /* red */
                color: white !important;
                border-radius: 6px !important;
                border: none !important;
            }
        """
    ):
        stop_button = st.button("Stop")

if stop_button:
    st.stop()

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
