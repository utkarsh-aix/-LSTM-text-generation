from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATA_PATH = DATA_DIR / "shakespeare.txt"

SEED = 42

# Quick settings designed for a ~45-minute interview assessment.
SEQUENCE_LENGTH = 20
MAX_SEQUENCES = 300_000
TRAIN_RATIO = 0.90

EMBEDDING_DIM = 128
LSTM_UNITS = 128
DROPOUT = 0.20
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

# Generation
DEFAULT_GENERATION_WORDS = 40
DEFAULT_TEMPERATURE = 0.8
