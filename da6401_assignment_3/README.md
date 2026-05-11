# DA6401 Assignment 3 - Transformer (German -> English)

**NAME:**: Harshit Shukla

**ROLL NO:** DA25S003

**GITHUB_LINK:** [https://github.com/harshit-8118/DA6401_Assignments/tree/master/da6401_assignment_3](https://github.com/harshit-8118/DA6401_Assignments/tree/master/da6401_assignment_3)

**WANDB_LINK:** [https://wandb.ai/da25s003-indian-institute-of-technology-madras/DA6401-Assignment_3/reports/Untitled-Report--VmlldzoxNjgzNDY4NA?accessToken=ij492cgciydjuwesv2bkypj7nf8qh1iax742w6awtzlfbbc3u0ez40iznciw1m23](https://wandb.ai/da25s003-indian-institute-of-technology-madras/DA6401-Assignment_3/reports/Untitled-Report--VmlldzoxNjgzNDY4NA?accessToken=ij492cgciydjuwesv2bkypj7nf8qh1iax742w6awtzlfbbc3u0ez40iznciw1m23)

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
python train.py --use-wandb --epochs 40 --batch-size 64 --scheduler noam --warmup-steps 4000 --label-smoothing 0.1 --checkpoint-path checkpoint.pt
```

## Resume training

```bash
python train.py --resume-from checkpoint.pt --epochs 30 --checkpoint-path checkpoint.pt
```

## Evaluation only (load a checkpoint then test BLEU)

```bash
python train.py --resume-from checkpoint.pt --eval-only
```