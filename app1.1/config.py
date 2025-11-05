from pathlib import Path

TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True, mode=0o755)
SENTIMENT_FILE_PATH = TEMP_DIR / "scraped_data.csv"
INAPROC_FILE_PATH = TEMP_DIR / "inaproc_data.csv"
PUTUSAN_MA_FILE_PATH = TEMP_DIR / "putusan_ma_data.csv"

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True, mode=0o755)
FEEDBACK_FILE_PATH = DATA_DIR / "feedback_data.csv"

NEWS_SOURCES = {
    'Kompas': 'kompas',
    'CNN Indonesia': 'cnn',
    'CNBC Indonesia': 'cnbc',
    'Kontan': 'kontan'
}
ADDITIONAL_SOURCES = {
    'INAPROC Daftar Hitam': 'inaproc',
    'Putusan MA': 'putusan_ma'
}
MODELS = {
    'indonesian-roberta-base': 'elidle/indonesian-roberta-base-company-reputation'
}

# Match this with your GPU (higher value = faster & more VRAM usage)
# You should set this to 1 if you are running on CPU
MODEL_BATCH_SIZE = 128
SENTIMENT_COLORS = {
    'positive': '#4CAF50',
    'neutral': '#9E9E9E',
    'negative': '#F44336',
}
