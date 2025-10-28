# src/inference.py
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

def predict(image_path, model_dir="outputs/model", tokenizer_dir="outputs/tokenizer"):
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    img = Image.open(image_path).convert("RGB")
    pixel_values = extractor(images=img, return_tensors="pt").pixel_values

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=32)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Predicted caption:", caption)

# пример запуска:
# python src/inference.py data/sample_images/COCO_val2014_000000391895.jpg
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Укажи путь к изображению")
        sys.exit()
    predict(sys.argv[1])
