# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os
from dataset import CocoCaptionsDataset
from model import build_model
from metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter

DATA_PATH = "data/caption_datasets/dataset_coco.json"
BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def train():
    model, tokenizer = build_model()
    dataset = CocoCaptionsDataset(DATA_PATH, image_root=None, split="test", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(dataloader)
    lr_scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

    writer = SummaryWriter(log_dir="logs")

    model.train()
    global_step = 0
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        # оценим метрики каждые 1 эпоху
        preds, refs = [], []
        model.eval()
        with torch.no_grad():
            for val_batch in dataloader:
                pixel_values = val_batch["pixel_values"].to(DEVICE)
                outputs = model.generate(pixel_values=pixel_values, max_length=32)
                pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refs_texts = val_batch["caption"]
                preds += pred_texts
                refs += refs_texts
                break  # одну партию достаточно для примера
        model.train()
        metrics = compute_metrics(preds, refs)
        print(f"Epoch {epoch+1} metrics:", metrics)
        for k, v in metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, epoch)

    writer.close()
    os.makedirs("outputs", exist_ok=True)
    model.save_pretrained("outputs/model")
    tokenizer.save_pretrained("outputs/tokenizer")

if __name__ == "__main__":
    train()
