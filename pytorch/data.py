import random

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from utils import get_line_offsets


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def __len__(self):
        return len(self.sp)

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, encoded):
        return self.sp.decode(encoded)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.offsets = get_line_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, 'r', encoding='utf-8') as file:
            file.seek(self.offsets[idx])
            text = file.readline().strip()
        ids = self.tokenizer.encode(text)
        return ids


class Collate:
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False, fold_size=None):
        assert not (pad_id < 0 and length_includes_pad)
        assert not (pad_id < 0 and fold_size)
        self.crop_length = crop_length
        self.fold_size = fold_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.pad_insert_rate = 0.0
        self.length_includes_pad = length_includes_pad

    def fold(self, ids):
        # pad the list for folding
        remainder = len(ids) % self.fold_size
        if remainder != 0:
            ids += [self.pad_id] * (self.fold_size - remainder)
        # fold the list
        ids = [ids[i:i + self.fold_size] for i in range(0, len(ids), self.fold_size)]
        return ids

    def generate_mask(self, length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    def process_ids(self, ids):
        # Add the eos token
        if self.eos_id >= 0:
            ids.append(self.eos_id)
        # Randomly insert pads into ids
        if self.pad_id >= 0 and self.pad_insert_rate > 0:
            pad_count = int(len(ids) * self.pad_insert_rate)
            pad_indices = random.sample(range(len(ids)), pad_count)
            for index in pad_indices:
                ids.insert(index, self.pad_id)
        if self.fold_size is not None:
            ids = self.fold(ids)
        # Crops the length
        if 0 < self.crop_length < len(ids):
            ids = ids[:self.crop_length]
        # Create a conditional mask
        conditional_mask = self.generate_mask(len(ids))
        return ids, len(ids), conditional_mask

    def __call__(self, batch):
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditional_mask = zip(*processed)

        # Sample a random amount of padding
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.length_includes_pad else torch.tensor(lengths)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_id
        )
        conditional_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditional_mask],
            batch_first=True,
            padding_value=False
        )

        return ids, lengths, conditional_mask
