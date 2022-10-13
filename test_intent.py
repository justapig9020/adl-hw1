import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
import pandas as pd

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

def max_index(tensor):
    return torch.max(tensor, -1)[1]

def main(args):
    device = args.device
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    for d in data:
        d['text'] = d['text'].split()
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    test_data = DataLoader(
        dataset,
        batch_size = args.batch_size,
        collate_fn = dataset.collate_fn,
        num_workers = args.num_workers)

    model = torch.load(args.ckpt_path)
    model.to(device)

    result = []
    for data in test_data:
        data['text'] = data['text'].to(device)
        id = data['id']
        output = model(data)
        pred = max_index(output)
        result += [{"id": i, "intent": dataset.idx2label(p.item())} for i, p in zip(id, pred)]
    # load weights into model
    df = pd.DataFrame(result)
    df.to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
