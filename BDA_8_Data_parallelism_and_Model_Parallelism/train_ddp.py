"""
Minimal PyTorch DDP training demo script for BDA_8.
This is a small, self-contained example that trains a tiny classifier
on a synthetic dataset using HuggingFace transformers (BERT) or any
AutoModelForSequenceClassification. It is intended as a runnable
proof-of-concept (does not train full BERT on large data).

Usage (inside the project's `.venv`):
  torchrun --nproc_per_node=2 train_ddp.py --model_name_or_path bert-base-uncased --epochs 1

Notes:
- Requires torch, transformers, datasets installed in the environment.
- For real FinBERT models replace --model_name_or_path with the desired HF model id.
"""
import argparse
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TinyTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_synthetic_text_data(n=200):
    texts = []
    labels = []
    for i in range(n):
        if random.random() < 0.5:
            texts.append('This is a positive example about finance and stocks.')
            labels.append(1)
        else:
            texts.append('This is a negative example about sports and entertainment.')
            labels.append(0)
    return texts, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    # DDP environment is expected to be set by torchrun
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    torch.distributed.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    texts, labels = create_synthetic_text_data(n=256)
    dataset = TinyTextDataset(texts, labels, tokenizer)

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if torch.distributed.get_rank() == 0:
            print(f'Epoch {epoch} loss {total_loss/len(loader):.4f}')

    # Save checkpoint only on rank 0
    if torch.distributed.get_rank() == 0:
        os.makedirs('checkpoints', exist_ok=True)
        model.module.save_pretrained('checkpoints/ddp_model')
        tokenizer.save_pretrained('checkpoints/ddp_model')
        print('Saved checkpoint to checkpoints/ddp_model')


if __name__ == '__main__':
    main()
