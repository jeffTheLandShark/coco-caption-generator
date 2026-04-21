import pickle
import torch
import os

from torch.utils.data import Dataset

from src.utils.config import *

class COCOCaptionDataset(Dataset):

    def __init__(self):
        # Load captions
        with open(CAPTIONS_FILE, "rb") as f:
            captions = pickle.load(f)

        # Load features - dict of {filename: feature_vector}
        # Rebuilding since features.pt is a flat tensor
        all_features = torch.load(FEATURES_FILE)

        # Get sorted filenames (same order extracted_features used)
        image_files = sorted([
            f for f in os.listdir(RAW_DIR + "images/")
            if f.endswith(".jpg")
        ])

        # Build filename -> feature vector lookup
        feature_dict = {
            fname: all_features[i]
            for i, fname in enumerate(image_files)
        }

        # Match each caption to its feature vector
        self.samples = []
        skipped = 0
        for rec in captions:
            fname = rec["file_name"]
            if fname not in feature_dict:
                skipped += 1
                continue
            self.samples.append({
                "feature": feature_dict[fname],
                "token_ids": torch.tensor(rec["token_ids"], dtype=torch.long),
                "caption": rec["caption"],
                "image_id": rec["image_id"],
            })
        
        print(f"[Dataset] {len(self.samples):,} samples loaded ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["feature"], sample["token_ids"]
