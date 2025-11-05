import torch
import streamlit as st
from transformers import pipeline
from transformers.pipelines.base import Pipeline
from datasets import load_dataset
import re
from pathlib import Path
import pandas as pd
import os
# Enable ROCm AOTriton and memory-efficient attention (important for AMD GPUs)
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["PYTORCH_ROCM_MEM_EFFICIENT_ATTENTION"] = "1"

@st.cache_resource(show_spinner=False, ttl=3600)
def load_sa_pipeline(
        model_path: str,
        batch_size: int = None
) -> Pipeline:
    """
    Loads sentiment analysis pipeline using Hugging Face library.
    Explicitly checks and prints device information.
    """
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = 0
        print("✅ Using Apple MPS GPU backend")
    else:
        device = -1
        print("⚠️ No GPU detected, using CPU")

    print(f"Pipeline will run on device index: {device}")

    return pipeline(
        'sentiment-analysis',
        model=model_path,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
        padding=True,
        device=device
    )


def analyze_sentiment(
    input_file: Path,
    sa_pipeline: Pipeline,
    pred_score_threshold: float = 0.70
) -> pd.DataFrame:
    """
    Analyze sentiment using Hugging Face model in sentiment analysis pipeline:
     - Uses Hugging Face dataset object for efficiency
     - Extracts all relevant sentences in title and body using regex
     - Passes them through the pipeline
     - Returns pandas DataFrame
    """
    #TODO: error checks when accessing column (e.g. row['query'] might raise KeyError)
    dataset = load_dataset('csv', data_files=str(input_file), split="train")
    batch_size = getattr(sa_pipeline, "_batch_size", None)
    if batch_size is None:
        batch_size = 1

    print("Analyzing titles...")
    title_pred = sa_pipeline(list(dataset['title']))
    dataset = dataset.add_column('title_pred', title_pred)

    print("Analyzing relevant sentences in bodies...")
    all_relevant_sentences = []

    for query, body in zip(dataset['query'], dataset['body']):
        query = str(query)
        body = str(body)

        sentences = re.split(r'(?<=[.!?])\s+', body)

        relevant_sentences = [
            sentence.strip()
            for sentence in sentences
            if re.search(r'\b' + re.escape(query) + r'\b', sentence, re.IGNORECASE)
        ]

        all_relevant_sentences.append(relevant_sentences)

    if any(all_relevant_sentences):
        flattened_sentences = [s for sublist in all_relevant_sentences for s in sublist]
        all_predictions = sa_pipeline(flattened_sentences, batch_size=batch_size)

        final_predictions = []
        start_idx = 0
        for relevant_sentences in all_relevant_sentences:
            preds = all_predictions[start_idx:start_idx + len(relevant_sentences)]
            final_predictions.append([
                {
                    'sentence': text,
                    'label': pred['label'],
                    'score': pred['score']
                }
                for text, pred in zip(relevant_sentences, preds)
            ])
            start_idx += len(relevant_sentences)
    else:
        final_predictions = [[] for _ in all_relevant_sentences]

    dataset = dataset.add_column('sentences_pred', final_predictions)

    print("Analyzing overall sentiment...")

    def get_predicted_labels(examples, pred_score_threshold):
        predicted_labels = []
        for title, query, title_pred, sentences_pred in zip(
                examples['title'], examples['query'], examples['title_pred'], examples['sentences_pred']
        ):
            label_weights = {'positive': 0, 'negative': 0, 'neutral': 0}

            if re.search(r'\b' + re.escape(query) + r'\b', title, re.IGNORECASE):
                if title_pred['score'] >= pred_score_threshold:
                    label_weights[title_pred['label']] += title_pred['score']

            for pred in sentences_pred:
                if pred['score'] >= pred_score_threshold:
                    label_weights[pred['label']] += pred['score']

            predicted_labels.append(max(label_weights.items(), key=lambda x: x[1])[0])

        return {'predicted_label': predicted_labels}

    dataset = dataset.map(
        get_predicted_labels,
        batched=True,
        fn_kwargs={'pred_score_threshold': pred_score_threshold}
    )

    print("✅ Sentiment analysis done successfully!")
    return dataset.to_pandas()
