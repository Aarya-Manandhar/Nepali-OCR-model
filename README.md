# Nepali OCR model

A CRNN with a CNN feature extractor, bidirectional LSTM, and CTC decoding for line-level Nepali OCR. The project includes a baseline checkpoint (`best.pt`) and a separate Hugging Face data fine-tune (`best_hf.pt`). The baseline remains the default because it performed better on the local `real_test` set.

## Recognize line images

Use Python 3.10 or newer. Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline checkpoint on one or more line images:

```bash
python predict.py path/to/line.png
```

Try the fine-tuned checkpoint explicitly:

```bash
python predict.py --checkpoint best_hf.pt path/to/line.png
```

The input should be a cropped, single-line image. `predict.py` runs on CPU.

## Fine-tune

Fine-tuning uses [himalaya-ai/devanagari_ocr_pretrain](https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_pretrain) and requires a CUDA-enabled PyTorch installation. First download/cache the dataset:

```bash
python -c "from datasets import load_dataset; load_dataset('himalaya-ai/devanagari_ocr_pretrain')"
```

Then run:

```bash
python finetune_hf.py
```

The script selects the newest cached snapshot, filters examples to single lines and characters supported by this model, and writes `best_hf.pt`. If a local `real_test/labels.txt` exists, it also reports CER for that set; test images and labels are not included in this repository.

## Evaluate CER

Prepare a folder with `labels.txt`, where each line contains a relative image path, a tab, and its transcription:

```text
img/line01.png<TAB>नेपाल राष्ट्र बैंक
```

Then run:

```bash
python eval_ours.py path/to/test_folder
```

The evaluator uses `best.pt`. The CER printed after fine-tuning in `finetune_hf.py` is for the selected fine-tuned checkpoint.

## Checkpoints

- `best.pt`: baseline model, default for inference.
- `best_hf.pt`: fine-tuned model; evaluate on your own data before choosing it for deployment.

Both checkpoints are PyTorch files containing model weights and the character vocabulary.
