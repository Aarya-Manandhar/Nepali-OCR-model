"""Fine-tune the existing CRNN on single-line labeled HF images."""
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import predict

ROOT = Path(__file__).resolve().parent
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/datasets--himalaya-ai--devanagari_ocr_pretrain/snapshots"
available_snapshots = list(SNAPSHOTS.glob("*")) if SNAPSHOTS.exists() else []
if not available_snapshots:
    raise FileNotFoundError(
        "No cached dataset snapshot found. Run the dataset download step in README.md first."
    )
SNAPSHOT = max(available_snapshots, key=lambda p: p.stat().st_mtime)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this fine-tuning run.")

DEVICE = torch.device("cuda")
HEIGHT, MAX_WIDTH, BATCH_SIZE = 48, 1536, 8
EPOCHS, LEARNING_RATE = 3, 3e-5
VOCAB = predict.vocab
CHARSET = set(VOCAB[1:])
BLANK = 0


def read_examples():
    examples = []
    for shard in sorted((SNAPSHOT / "ocr_shards").glob("*.ocr.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = re.sub(r"\s+", " ", row["ocr"]).strip()
                if not (2 <= len(text) <= 80) or any(c not in CHARSET for c in text):
                    continue
                if "\n" in row["ocr"] or "\r" in row["ocr"]:
                    continue
                path = SNAPSHOT / row["image"]
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                h, w = image.shape
                if h > 140:
                    continue
                width = min(MAX_WIDTH, max(16, int(w * HEIGHT / h) // 4 * 4))
                repeated = sum(a == b for a, b in zip(text, text[1:]))
                if width // 4 < len(text) + repeated:
                    continue
                examples.append((str(path), text))
    return examples


class OCRDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.index = {char: i for i, char in enumerate(VOCAB) if i > 0}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        path, text = self.examples[i]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        width = min(MAX_WIDTH, max(16, int(w * HEIGHT / h) // 4 * 4))
        image = cv2.resize(image, (width, HEIGHT)).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(image)[None], [self.index[c] for c in text], text


def collate(batch):
    width = max(item[0].shape[-1] for item in batch)
    images = torch.ones(len(batch), 1, HEIGHT, width)
    for i, (image, _, _) in enumerate(batch):
        images[i, :, :, : image.shape[-1]] = image
    targets = torch.tensor([char for _, seq, _ in batch for char in seq], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for _, seq, _ in batch], dtype=torch.long)
    input_lengths = torch.tensor([image.shape[-1] // 4 for image, _, _ in batch], dtype=torch.long)
    return images, targets, input_lengths, target_lengths, [text for _, _, text in batch]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def score(model, loader):
    model.eval()
    errors = total = 0
    with torch.no_grad():
        for images, _, _, _, truths in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images.to(DEVICE, non_blocking=True))
            predictions = predict.decode(logits.float())
            for prediction, truth in zip(predictions, truths):
                errors += edit_distance(prediction, truth)
                total += len(truth)
    model.train()
    return errors / max(total, 1)


def real_test_score(model):
    rows = []
    labels_path = ROOT / "real_test/labels.txt"
    if not labels_path.exists():
        print("real_test/labels.txt not found; skipping real_test CER.", flush=True)
        return None
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                name, truth = line.rstrip("\n").split("\t", 1)
                rows.append((ROOT / "real_test" / name, truth))
    errors = total = 0
    model.eval()
    with torch.no_grad():
        for path, truth in rows:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            width = min(MAX_WIDTH, max(16, int(w * HEIGHT / h) // 4 * 4))
            image = cv2.resize(image, (width, HEIGHT)).astype(np.float32) / 127.5 - 1
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(torch.from_numpy(image)[None, None].to(DEVICE))
            prediction = predict.decode(logits.float())[0]
            errors += edit_distance(prediction, truth)
            total += len(truth)
    return errors / max(total, 1)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    examples = read_examples()
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.1))
    validation, training = examples[:split], examples[split:]
    print(f"HF single-line pairs: {len(examples)}; train: {len(training)}; validation: {len(validation)}", flush=True)

    train_loader = DataLoader(OCRDataset(training), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(OCRDataset(validation), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate, pin_memory=True)
    model = predict.model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda")
    best_val = float("inf")
    output = ROOT / "best_hf.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, (images, targets, input_lengths, target_lengths, _) in enumerate(train_loader, 1):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            input_lengths = input_lengths.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            log_probs = logits.float().log_softmax(2).transpose(0, 1)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            if step % 100 == 0:
                print(f"epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} loss {running/100:.4f}", flush=True)
                running = 0.0

        val_cer = score(model, val_loader)
        print(f"epoch {epoch}: validation CER {val_cer:.4f}", flush=True)
        if val_cer < best_val:
            best_val = val_cer
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "vocab": VOCAB, "epoch": epoch, "validation_cer": val_cer,
                        "dataset": "himalaya-ai/devanagari_ocr_pretrain"}, output)
            print(f"saved {output.name}", flush=True)

    checkpoint = torch.load(output, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_cer = real_test_score(model)
    print(f"best validation CER: {best_val:.4f}", flush=True)
    if test_cer is not None:
        print(f"fine-tuned real_test CER: {test_cer:.4f}", flush=True)


if __name__ == "__main__":
    main()
