# predict.py -- read Nepali text lines with the trained model (CPU only)
# usage: python predict.py [--checkpoint best.pt] img1.jpg img2.jpg ...
import argparse
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn

H, MAXW = 48, 1536

ROOT = Path(__file__).resolve().parent
ck = torch.load(ROOT / "best.pt", map_location="cpu", weights_only=False)
vocab = ck["vocab"]

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class CRNN(nn.Module):
    def __init__(self, nc, rnn_h=384, layers=3):
        super().__init__()
        self.cnn = nn.Sequential(
            block(1, 64), block(64, 64), nn.MaxPool2d(2, 2),
            block(64, 128), block(128, 128), nn.MaxPool2d(2, 2),
            block(128, 256), block(256, 256), nn.MaxPool2d((2, 1)),
            block(256, 512), block(512, 512), nn.MaxPool2d((2, 1)))
        self.proj = nn.Conv2d(512, 512, (3, 1))
        self.rnn = nn.LSTM(512, rnn_h, layers, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(2 * rnn_h, nc)
    def forward(self, x):
        f = self.proj(self.cnn(x)).squeeze(2).permute(0, 2, 1)
        return self.fc(self.rnn(f)[0])

model = CRNN(len(vocab))
model.load_state_dict(ck["model"])
model.eval()

def decode(logits):
    out = []
    for seq in logits.argmax(-1).tolist():
        s, last = [], 0
        for k in seq:
            if k != last and k != 0:
                s.append(vocab[k])
            last = k
        out.append("".join(s))
    return out

def read_line(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    nw = min(MAXW, max(16, int(w * H / h) // 4 * 4))
    x = cv2.resize(img, (nw, H)).astype(np.float32) / 127.5 - 1
    with torch.no_grad():
        return decode(model(torch.from_numpy(x)[None, None]))[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize Nepali text in line images")
    parser.add_argument("--checkpoint", default="best.pt", help="model checkpoint (default: best.pt)")
    parser.add_argument("images", nargs="+", help="paths to line images")
    args = parser.parse_args()
    selected = Path(args.checkpoint)
    if not selected.is_absolute():
        selected = ROOT / selected
    ck = torch.load(selected, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model.load_state_dict(ck["model"])
    model.eval()
    for p in args.images:
        print(p, "->", read_line(p))
