import json
import os
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoFeatureExtractor

class CocoCaptionsDataset(Dataset):
    def __init__(self, json_path, image_root=None, split="train",
                 feature_extractor_name="google/vit-base-patch16-224-in21k", tokenizer=None, max_length=32):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = [img for img in data["images"] if img["split"] == split or split == "all"]
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        caption = sample["sentences"][0]["raw"].strip()
        filename = sample["filename"]
        image_path = os.path.join(self.image_root, filename) if self.image_root else None

        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="white")  # если нет картинок

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]

        if self.tokenizer:
            labels = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)
        else:
            labels = caption

        return {"pixel_values": pixel_values, "labels": labels, "caption": caption, "filename": filename}
