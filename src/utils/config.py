# Shared configuration — update ONLY with team agreement

# Model
EMBED_DIM = None
HIDDEN_DIM = None
FEATURE_DIM = None

# Data
MAX_SEQ_LEN = None
BATCH_SIZE = None
VOCAB_MIN_FREQ = None

# Training
LEARNING_RATE = None
NUM_EPOCHS = None

# Paths
DATA_DIR = "data/"
RAW_DIR = DATA_DIR + "raw/"
PROCESSED_DIR = DATA_DIR + "processed/"
FEATURE_DIR = DATA_DIR + "features/"

VOCAB_FILE = PROCESSED_DIR + "vocab.json"
CAPTIONS_FILE = PROCESSED_DIR + "captions.pkl"
FEATURES_FILE = FEATURE_DIR + "features.pt"