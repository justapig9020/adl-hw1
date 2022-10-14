import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

ACCURACY = 'acc'
LOSS = 'loss'
ITER = 'iter'
TOKEN_ACC = 'token_acc'

class TrainingLogger:
    def __init__(self, threshold):
        self.log = {
            TRAIN: {
                ACCURACY: [],
                LOSS: [],
                TOKEN_ACC: [],
                ITER: [],
            },
            DEV: {
                ACCURACY: [],
                LOSS: [],
                TOKEN_ACC: [],
                ITER: [],
            }
        }
        self.best_loss = None
        self.non_decrease_cnt = 0
        self.threshold = threshold
        self.new_record = False
    def record(self, type: str, result: Dict, iter: int):
        self.log[type][ACCURACY].append(result[ACCURACY])
        self.log[type][LOSS].append(result[LOSS])
        self.log[type][ITER].append(iter)
        self.log[type][TOKEN_ACC].append(result[TOKEN_ACC])
        if type == DEV:
            if self.best_loss is None or result[LOSS] >= self.best_loss:
                self.best_loss = result[LOSS]
                self.non_decrease_cnt += 1
            else:
                self.new_record = True
                if self.non_decrease_cnt > 0:
                    self.non_decrease_cnt -= 1
    def is_new_record(self) -> bool:
        ret = self.new_record
        self.new_record = False
        return ret
    def early_return(self) -> bool:
        return self.non_decrease_cnt >= self.threshold

def plotter(name, logger: TrainingLogger):
    targets = [LOSS, ACCURACY, TOKEN_ACC]
    axs = (plt.figure(constrained_layout = True, figsize=(12, 6)).subplots(1, len(targets), sharex = False, sharey = False))
    for target, ax in zip(targets, axs):
        if 'acc' in target:
            ax.set_ylim([0, 1])
        ax.set_title(target)
        ax.set_ylabel(target.lower())
        ax.set_xlabel('epoch')
        ax.plot(logger.log[TRAIN][ITER], logger.log[TRAIN][target], '-', color = (1, 100 / 255, 100/ 255), label = 'Train')
        ax.plot(logger.log[DEV][ITER], logger.log[DEV][target], '--', color = (100 / 255, 1, 100/ 255), label = 'Eval')
        ax.legend()
    plt.savefig(f"./plot/slot/{name}.png")
    plt.cla()
    plt.clf()
    plt.close()

def print_log(train, eval):
    print(f'Train: accuracy {train[ACCURACY]:.03f}, loss {train[LOSS]:.03f}')
    print(f'Eval: accuracy {eval[ACCURACY]:.03f}, loss {eval[LOSS]:.03f}')

def init_dataset(args):
    with open(args.cache_dir / 'vocab.pkl', 'rb') as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / 'tag2idx.json'
    tag2idx = json.loads(tag_idx_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    train_data = DataLoader(
        dataset = datasets[TRAIN],
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        collate_fn = datasets[TRAIN].collate_fn)
    eval_data = DataLoader(
        dataset = datasets[DEV],
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        collate_fn = datasets[DEV].collate_fn)
    return datasets[TRAIN].num_classes + 1, train_data, eval_data

def init_model(args, num_classes):
    embeddings = torch.load(args.cache_dir / 'embeddings.pt')
    model = SeqTagger(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = num_classes
    )
    return model

def max_index(tensor):
    return torch.max(tensor, -1)[1]

def token_correct_count(classes, label):
    return (classes == label).float().sum().item()

def correct_count(classes, label):
    seqs = torch.all(classes == label, dim = -1)
    return seqs.float().sum().item()

def do_epoch(device, model, dataset, loss_fn, optimizer = None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    token_acc = 0
    correct = 0
    total_seq = 0
    total_token = 0
    sum_loss = 0.0
    for data in dataset:
        label = data['tags'].to(device)
        batch_size = label.shape[0]
        seq_len = label.shape[1]
        data['tokens'] = data['tokens'].to(device)
        # Forward
        output = model(data)
        # Loss
        reshape_output = torch.reshape(output, (-1, model.num_class))
        reshape_label = torch.reshape(label, (-1, ))
        loss = loss_fn(reshape_output, reshape_label)

        sum_loss += loss.item() * batch_size

        if training:
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
        
        classes = max_index(output)
        correct += correct_count(classes, label)
        token_acc += token_correct_count(classes, label)
        total_seq += batch_size
        total_token += batch_size * seq_len
    result = {
        LOSS: sum_loss / total_seq,
        ACCURACY: correct / total_seq,
        TOKEN_ACC: token_acc / total_token,
    }
    return result

def main(args):
    device = args.device
    model_name = f"{args.name}_{args.hidden_size}_{args.num_layers}_{args.dropout}_{args.lr}"
    save_path = args.ckpt_dir / f"{model_name}.pt"

    # Initialization
    num_classes, train_data, eval_data = init_dataset(args)

    model = init_model(args, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = 1)

    model.to(device)
    loss_fn.to(device)

    # Init loggers
    logger = TrainingLogger(args.early_return)

    # Start training
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for i, epoch in enumerate(epoch_pbar):
        train_result = do_epoch(
            device = device,
            model = model,
            dataset = train_data,
            loss_fn = loss_fn,
            optimizer = optimizer, 
        )
        logger.record(TRAIN, train_result, i)

        eval_result = do_epoch(
            device = device,
            model = model,
            dataset = eval_data,
            loss_fn = loss_fn,
        )
        logger.record(DEV, eval_result, i)
        if logger.is_new_record():
            torch.save(model, save_path)

        if i % 5 == 0:
            print_log(train_result, eval_result)
            plotter(model_name, logger)
            print(f'token acc {train_result["token_acc"] = }, {eval_result["token_acc"] = }')
        if logger.early_return():
            return


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
    parser.add_argument(
        "--name",
        type=str,
        default=""
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--early_return", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)