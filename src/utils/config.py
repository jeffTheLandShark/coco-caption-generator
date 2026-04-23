# Shared configuration for model architecture

# Model dimensions
EMBED_DIM = 256
HIDDEN_DIM = 512
FEATURE_DIM = 2048

# Data
MAX_SEQ_LEN = 20
BATCH_SIZE = 32
VOCAB_MIN_FREQ = 5

# Training
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# Paths
DATA_DIR = "data/"
RAW_DIR = DATA_DIR + "raw/"
PROCESSED_DIR = DATA_DIR + "processed/"
FEATURE_DIR = DATA_DIR + "features/"
SRC_DIR = "src/"
TRAIN_DIR = SRC_DIR + "train/"

# Files
VOCAB_FILE = PROCESSED_DIR + "vocab.json"
CAPTIONS_FILE = PROCESSED_DIR + "captions.pkl"
FEATURES_FILE = FEATURE_DIR + "features.pt"
MODEL_FILE = TRAIN_DIR + "trained_model.pt"