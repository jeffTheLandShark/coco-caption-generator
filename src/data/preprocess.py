import os
import json
import re
import pickle
from collections import Counter

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.vocab import Vocabulary
from src.utils.config import *


def load_captions():
    """
    Load raw COCO captions.

    Returns:
        list[str]
    """
    ann_file = RAW_DIR + "annotations/captions_val2017.json"
    
    with open(ann_file) as f:
        data = json.load(f)
    
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    records = []
    missing = 0
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        file_name = id_to_filename.get(img_id)
        if file_name is None:
            missing += 1
            continue
        
        img_path = RAW_DIR + "images/" + file_name
        if not os.path.exists(img_path):
            missing += 1
            continue

        caption = re.sub(r"[^a-z0-9\s]", "", ann["caption"].lower()).strip()
        if not caption:
            continue
        
        records.append({
            "image_id": img_id,
            "file_name": file_name,
            "caption": caption,
        })

    print(f"[Preprocess] Kept {len(records):,} records ({missing:,} skipped)")
    return records


def build_vocab(captions, min_freq=VOCAB_MIN_FREQ):
    """
    Build vocabulary from captions.

    Returns:
        dict[str, int]
    """
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(captions)
    return vocab


def encode_captions(captions, vocab):
    """
    Convert captions to padded integer sequences.

    Returns:
        list[list[int]]
    """
    encoded = []
    for rec in captions:
        token_ids = vocab.encode(rec["caption"])

        # truncate if over max length
        token_ids = token_ids[:MAX_SEQ_LEN]

        # pad if under max length
        token_ids += [vocab.pad_idx] * (MAX_SEQ_LEN - len(token_ids))

        encoded.append({
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "caption": rec["caption"],
            "token_ids": token_ids,
        })
    
    return encoded


def main():
    """
    End-to-end preprocessing:
    - load captions
    - build vocab
    - encode captions
    - save outputs
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load 
    records = load_captions()

    # 2. Build vocab
    vocab = build_vocab([r["caption"] for r in records])
    vocab.save(VOCAB_FILE)

    # 3. Encode
    encoded = encode_captions(records, vocab)

    # 4. Save captions as pickle
    with open(CAPTIONS_FILE, "wb") as f:
        pickle.dump(encoded, f)
    print(f"[Preprocess] Captions saved -> {CAPTIONS_FILE}")

    # Verification
    sample = encoded[0]
    print(f"\n[Preprocess] Sample:")
    print(f" image_id  : {sample['image_id']}")
    print(f" caption   : {sample['caption']}")
    print(f" token_ids : {sample['token_ids']}")
    print(f" decoded   : {vocab.decode(sample['token_ids'])}")

if __name__ == "__main__":
    main()