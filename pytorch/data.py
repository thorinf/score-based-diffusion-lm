from collections import deque
from logging import getLogger
import random
from typing import List

import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset

from utils import get_line_offsets

logger = getLogger()


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.num_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(f"Vocab Size: {self.num_words:,}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def __len__(self):
        return self.num_words

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        encoded = self.sp_model.encode(text)
        if bos:
            encoded = [self.bos_id] + encoded
        if eos:
            encoded = encoded + [self.eos_id]
        return encoded

    def decode(self, encoded) -> str:
        return self.sp_model.decode(encoded)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer):
        self.path = path
        self.tokenizer = tokenizer

        logger.info(f"Finding lines in large text file {path}")
        self.offsets = get_line_offsets(path)
        logger.info(f"Found {len(self.offsets):,} lines in {path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, 'r', encoding='utf-8') as file:
            file.seek(self.offsets[idx])
            text = file.readline().strip()
        ids = self.tokenizer.encode(text, bos=True, eos=True)
        return ids


class PackedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, sizing_fn, packed_collate_fn, **kwargs):
        self.sizing_fn = sizing_fn
        self.packed_collate_fn = packed_collate_fn
        super(PackedDataLoader, self).__init__(*args, **kwargs)
        self.cache = deque()

    def pack_samples(self, samples):
        packed_indexes = pack_sizes(self.sizing_fn(samples))
        return [[samples[index] for index in sublist] for sublist in packed_indexes]

    def __iter__(self):
        for samples in super(PackedDataLoader, self).__iter__():
            self.cache.extend(self.pack_samples(samples))
            while len(self.cache) >= self.batch_size:
                yield self.packed_collate_fn([self.cache.popleft() for _ in range(self.batch_size)])


class Collate:
    def __init__(
            self,
            max_sequence_length: int = -1,
            pad_sequence_value: int = 0,
            random_length_expansion: bool = False,
            insert_value: int = -1,
            insert_rate: float = 0.0
    ) -> None:
        assert not (random_length_expansion and pad_sequence_value < 0)
        assert not (insert_rate > 0.0 and insert_value < 0)
        self.max_sequence_length = max_sequence_length
        self.pad_sequence_value = pad_sequence_value
        self.random_length_expansion = random_length_expansion
        self.insert_value = insert_value
        self.insert_rate = insert_rate

    def collate_fn(self, batch):
        processed = list(map(self._process_ids, batch))
        ids, lengths, conditioning_mask = zip(*processed)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_sequence_value
        )
        conditioning_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditioning_mask],
            batch_first=True,
            padding_value=False
        )

        length_mask = torch.lt(torch.arange(ids.shape[1]).unsqueeze(0), torch.tensor(lengths).unsqueeze(1))

        return ids, length_mask, conditioning_mask

    def prepack_fn(self, batch):
        return list(map(self._process_ids, batch))

    @staticmethod
    def sizing_fn(batch):
        return [sample[1] for sample in batch]

    def packed_collate_fn(self, batch):
        processed = list(map(self._process_packed_samples, batch))
        ids, lengths, attention_masks, conditioning_masks = zip(*processed)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_sequence_value
        )

        length_mask = torch.lt(torch.arange(ids.shape[1]).unsqueeze(0), torch.tensor(lengths).unsqueeze(1))

        max_size = max(len(mask) for mask in attention_masks)

        def pad_tensor(t, target_size):
            padding_size = target_size - t.size(0)
            return torch.nn.functional.pad(t, (0, padding_size, 0, padding_size))

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [pad_tensor(torch.tensor(mask, dtype=torch.bool), max_size) for mask in attention_masks],
            batch_first=True,
            padding_value=False
        ).unsqueeze(1)

        conditioning_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditioning_masks],
            batch_first=True,
            padding_value=False
        )

        return ids, length_mask, attention_mask, conditioning_mask

    def _process_ids(self, ids):
        ids = self._crop_augment_ids(ids)
        conditioning_mask = self._gen_conditioning_mask(len(ids))
        return ids, len(ids), conditioning_mask

    def _crop_augment_ids(self, ids):
        if self.insert_value >= 0 and self.insert_rate > 0.0:
            ids = self._pad_insert(ids)
        if 0 < self.max_sequence_length < len(ids):
            ids = self._random_crop(ids)
        elif self.random_length_expansion and len(ids) < self.max_sequence_length:
            ids = self._expand_length(ids)
        return ids

    def _pad_insert(self, ids):
        # Randomly insert into ids
        pad_count = int(len(ids) * self.insert_rate)
        pad_indices = random.sample(range(len(ids)), pad_count)
        for index in pad_indices:
            ids.insert(index, self.insert_value)
        return ids

    def _random_crop(self, ids):
        # Random segment of full sequence if too long
        start_index = random.randint(0, len(ids) - self.max_sequence_length)
        end_index = start_index + self.max_sequence_length
        ids = ids[start_index:end_index]
        return ids

    def _expand_length(self, ids):
        # Sample a random amount of padding
        max_padding_length = self.max_sequence_length - len(ids)
        random_padding_length = random.randint(0, max_padding_length)
        ids.extend([self.pad_sequence_value] * random_padding_length)
        return ids

    @staticmethod
    def _gen_conditioning_mask(length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # Half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    @staticmethod
    def _process_packed_samples(packed_samples):
        packed_ids, packed_lengths, packed_conditioning_masks = zip(*packed_samples)

        ids = [item for sublist in packed_ids for item in sublist]
        flattened_conditioning_mask = [item for sublist in packed_conditioning_masks for item in sublist]
        attention_mask = packed_mask(packed_lengths)
        return ids, sum(packed_lengths), attention_mask, flattened_conditioning_mask


def pack_sizes(sizes):
    max_size = max(sizes)
    sorted_indices = sorted(range(len(sizes)), key=lambda k: sizes[k], reverse=True)
    bins, bin_sums = [], {}

    for index in sorted_indices:
        for b, total in bin_sums.items():
            if total + sizes[index] <= max_size:
                bins[b].append(index)
                bin_sums[b] += sizes[index]
                break
        else:
            bins.append([index])
            bin_sums[len(bins) - 1] = sizes[index]

    return bins


def packed_mask(lengths):
    mask = []
    for l in lengths:
        row = []
        for ll in lengths:
            row.extend([1 if ll == l else 0] * ll)
        mask.extend([row] * l)
    return mask


def infinite_loader(dataloader: torch.utils.data.DataLoader):
    while True:
        for batch in dataloader:
            yield batch
