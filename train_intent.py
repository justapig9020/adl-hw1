import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import trange

from model import SeqClassifier
from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

ACCURACY = 'acc'
LOSS = 'loss'
ITER = 'iter'

def spliter(data):
    for d in data:
        d['text'] = d['text'].split()
    return data

def init_dataset(args: Namespace) -> Tuple[int, DataLoader, DataLoader]:
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    data = {split: spliter(slice_data) for split, slice_data in data.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
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
    return datasets[TRAIN].num_classes, train_data, eval_data

def init_model(args: Namespace, num_classes: int) -> SeqClassifier:
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        rnn = args.rnn,
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = num_classes
    )
    return model

class TrainingLogger:
    def __init__(self, threshold):
        self.log = {
            TRAIN: {
                ACCURACY: [],
                LOSS: [],
                ITER: [],
            },
            DEV: {
                ACCURACY: [],
                LOSS: [],
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
    targets = [LOSS, ACCURACY]
    for target in targets:
        plt.title(target)
        plt.ylabel(target.lower())
        plt.xlabel('epoch')
        plt.plot(logger.log[TRAIN][ITER], logger.log[TRAIN][target], '-', color = (1, 100 / 255, 100/ 255))
        plt.plot(logger.log[DEV][ITER], logger.log[DEV][target], '--', color = (100 / 255, 1, 100/ 255))
        plt.savefig(f"./plot/intent/{name}_{target}.png")
        plt.clf()
        plt.cla()

def print_log(train, eval):
    print(f'Train: accuracy {train[ACCURACY]:.03f}, loss {train[LOSS]:.03f}')
    print(f'Eval: accuracy {eval[ACCURACY]:.03f}, loss {eval[LOSS]:.03f}')

def max_index(tensor):
    return torch.max(tensor, -1)[1]

def correct_count(classes, label):
    return (classes == label).float().sum().item()

def do_epoch(device, model, dataset: DataLoader, loss_fn, optimizer = None) -> Dict:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    correct = 0
    total = 0
    sum_loss = 0.0
    for data in dataset:
        label = data['intent'].to(device)
        batch_size = label.shape[0]
        data['text'] = data['text'].to(device)
        # Forward
        output = model(data)
        # Loss
        loss = loss_fn(output, label) 

        sum_loss += loss.item() * batch_size

        if training:
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
        
        classes = max_index(output)
        correct += correct_count(classes, label)
        total += batch_size
    result = {
        LOSS: sum_loss / total,
        ACCURACY: correct / total,
    }
    return result

def main(args: Namespace):
    device = args.device
    model_name = f"{args.name}_{args.hidden_size}_{args.num_layers}_{args.dropout}"
    save_path = args.ckpt_dir / f"{model_name}.pt"

    # Initialization
    num_classes, train_data, eval_data = init_dataset(args)

    model = init_model(args, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

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
            if logger.early_return():
                return



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--name",
        type=str,
        default='',
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn", type=str, default="LSTM")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

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
