from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        """
        The data will end up with 3 ~ 4 keys: 
            - 'text'
            - 'len'
            - 'intent'
            - 'id'
        Each of the keys contents a list or tensor that contents data of each training / evaluation instances
        """
        data = {} 

        texts = [sample['text'] for sample in samples]
        # Encode texts to corresponding id and do padding
        data['text'] = torch.tensor(self.vocab.encode_batch(texts))
        data['len'] = torch.tensor([len(text) for text in texts])
        data['id'] = [sample['id'] for sample in samples]

        # Since testing data does not content 'intent', we have to check for it
        if 'intent' in samples[0].keys():
            labels = [sample['intent'] for sample in samples]
            data['intent'] = torch.tensor([self.label2idx(label) for label in labels])
        return data


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        if idx in self._idx2label.keys():
            return self._idx2label[idx]
        else:
            return '[UNK]'


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        """
        The data will end up with 3 ~ 4 keys: 
            - 'tokens'
            - 'len'
            - 'tags'
            - 'id'
        Each of the keys contents a list or tensor that contents data of each training / evaluation instances
        """
        data = {} 

        tokens = [sample['tokens'] for sample in samples]
        # Encode texts to corresponding id and do padding
        data['tokens'] = torch.tensor(self.vocab.encode_batch(tokens))
        data['len'] = torch.tensor([len(token) for token in tokens])
        data['id'] = [sample['id'] for sample in samples]

        # Since testing data does not content 'intent', we have to check for it
        if 'tags' in samples[0].keys():
            tags = [sample['tags'] for sample in samples]
            to_len = max(data['len'])
            tags = [[self.label2idx(t) for t in tag] for tag in tags]
            data['tags'] = torch.tensor(pad_to_len(tags, to_len, self.num_classes))
        return data