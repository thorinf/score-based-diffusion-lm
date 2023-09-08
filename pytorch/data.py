import random
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor
import torch
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

    @staticmethod
    def generate_conditioning_mask(length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # Half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    def process_ids(self, ids):
        # Randomly insert into ids
        if self.insert_value >= 0 and self.insert_rate > 0.0:
            pad_count = int(len(ids) * self.insert_rate)
            pad_indices = random.sample(range(len(ids)), pad_count)
            for index in pad_indices:
                ids.insert(index, self.insert_value)

        # Random segment of full sequence if too long
        if 0 < self.max_sequence_length < len(ids):
            start_index = random.randint(0, len(ids) - self.max_sequence_length)
            end_index = start_index + self.max_sequence_length
            ids = ids[start_index:end_index]

        # Create a conditional mask
        conditioning_mask = self.generate_conditioning_mask(len(ids))

        return ids, len(ids), conditioning_mask

    def __call__(self, batch):
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditioning_mask = zip(*processed)

        # Sample a random amount of padding
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.random_length_expansion else torch.tensor(lengths)

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

        length_mask = torch.lt(torch.arange(ids.shape[1]).unsqueeze(0), lengths.unsqueeze(1))

        return ids, length_mask, conditioning_mask
