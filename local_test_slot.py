import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

def max_index(tensor):
    return torch.max(tensor, -1)[1]

def correct_count(classes, label):
    seqs = torch.all(classes == label, dim = -1)
    return seqs.float().sum().item()

def main(args):
    device = args.device

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, collate_fn=dataset.collate_fn)
    path = args.ckpt_dir
    print(path)
    model = torch.load(path)
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    for data in test_loader:
        label = data['tags'].to(device)
        batch_size = label.shape[0]
        seq_len = label.shape[1]
        data['tokens'] = data['tokens'].to(device)
        output = model(data)

        classes = max_index(output)
        correct += correct_count(classes, label)
        total += batch_size
    print(f"Accuracy: {correct / total}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")
    parser.add_argument("--test_file", type=Path, required=True)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)