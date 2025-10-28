import json, os, random, shutil

DATA_DIR = "data"
JSON_PATH = os.path.join(DATA_DIR, "caption_datasets", "dataset_coco.json")
OUT_JSON = os.path.join(DATA_DIR, "karpathy_sample.json")
OUT_IMG_DIR = os.path.join(DATA_DIR, "sample_images")
SRC_IMG_DIRS = [os.path.join(DATA_DIR, "train2014"), os.path.join(DATA_DIR, "val2014")]
N = 1000  # сколько изображений в sample

os.makedirs(OUT_IMG_DIR, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# предположим data["images"] существет
images = data.get("images", [])
if len(images) == 0:
    raise SystemExit("Не найден раздел images в JSON")

sample = random.sample(images, min(N, len(images)))

new_images = []
for img in sample:
    # определяем возможный файл_path
    file_path = img.get("file_path") or img.get("filename") or img.get("coco_url", "").split("/")[-1]
    # пробуем найти файл в train/val
    found = False
    for src in SRC_IMG_DIRS:
        src_path = os.path.join(src, file_path)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(OUT_IMG_DIR, file_path))
            found = True
            break
    if found:
        new_images.append(img)
    else:
        # если не нашли — пропускаем
        continue

new_data = {"images": new_images}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False)
print(f"Создан sample: {OUT_JSON}, images -> {len(new_images)}")
