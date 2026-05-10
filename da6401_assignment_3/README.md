# DA6401 Assignment 3 - Transformer (German -> English)

This repo now contains a complete training pipeline for Multi30k using:
- Hugging Face dataset loader: `load_dataset("bentrevett/multi30k")`
- spaCy tokenization (`spacy.blank("de")`, `spacy.blank("en")`)
- Transformer model from scratch (`model.py`)
- Label smoothing, Noam scheduler, greedy decoding, BLEU evaluation, checkpoints
- Full CLI via `argparse` in `train.py`

## Files

```text
da6401_assignment_3/
|-- dataset.py
|-- lr_scheduler.py
|-- model.py
|-- train.py
|-- requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Train (Noam scheduler + label smoothing)

```bash
python train.py --use-wandb --epochs 20 --batch-size 64 --scheduler noam --warmup-steps 4000 --label-smoothing 0.1 --checkpoint-path checkpoint.pt
```

## Train without W&B

```bash
python train.py --epochs 20 --batch-size 64 --scheduler noam --warmup-steps 4000 --label-smoothing 0.1 --checkpoint-path checkpoint.pt
```

## Resume training

```bash
python train.py --resume-from checkpoint.pt --epochs 30 --checkpoint-path checkpoint.pt
```

## Fixed learning rate baseline

```bash
python train.py --scheduler fixed --lr 1e-4 --epochs 20 --checkpoint-path fixed_lr.pt
```

## Evaluation only (load a checkpoint then test BLEU)

```bash
python train.py --resume-from checkpoint.pt --eval-only
```

## Quick smoke run (small subset)

```bash
python train.py --epochs 1 --batch-size 8 --d-model 128 --num-layers 2 --num-heads 4 --d-ff 512 --max-train-samples 256 --max-val-samples 64 --max-test-samples 64
```
