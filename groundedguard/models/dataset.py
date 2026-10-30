"""Module 3a: one training example = ONE RESPONSE with ALL its context windows.

The windows of a response are run through the encoder together, and their response-
token scores are MIN-aggregated (groundedguard/models/aggregate.py) before the loss. So a supported
token only has to look supported in the window that holds its evidence, and every
response counts once in the loss, however many windows it has.

Usage:
    ds = GroupedWindowDataset(PROCESSED_DIR / "train_windows.pkl")
    sampler = WindowBudgetSampler(ds.n_windows, max_windows=32, shuffle=True)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=Collator(tok.pad_token_id))
"""

import random

import torch
from torch.utils.data import Dataset, Sampler

from groundedguard.data.io import load_pickle

IGNORE = -100


class GroupedWindowDataset(Dataset):
    def __init__(self, path):
        data = load_pickle(path)
        self.config = data["config"]
        self.responses = data["responses"]
        by_id: dict[str, list[dict]] = {}
        for w in data["windows"]:
            by_id.setdefault(w["id"], []).append(w)
        self.ids = list(by_id)
        self.windows = [sorted(by_id[i], key=lambda w: w["window"]) for i in self.ids]
        self.n_windows = [len(ws) for ws in self.windows]
        self.window_lens = [[len(w["input_ids"]) for w in ws] for ws in self.windows]

    def subset(self, n: int, seed: int) -> None:
        """Keep a random n responses, in place (smoke tests)."""
        keep = sorted(random.Random(seed).sample(range(len(self.ids)), min(n, len(self.ids))))
        self.ids = [self.ids[i] for i in keep]
        self.windows = [self.windows[i] for i in keep]
        self.n_windows = [self.n_windows[i] for i in keep]
        self.window_lens = [self.window_lens[i] for i in keep]
        self.responses = {i: self.responses[i] for i in self.ids}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        rid = self.ids[idx]
        return {
            "id": rid,
            "input_ids": [w["input_ids"] for w in self.windows[idx]],
            "response_start": [w["response_start"] for w in self.windows[idx]],
            "labels": self.responses[rid]["token_labels"],
        }


class Collator:
    """Flattens the windows of all responses in the batch into one encoder batch.

    Returns
        input_ids, attention_mask  [W, L]   all windows, padded
        group, slot                [W]      which response / which of its windows
        response_start             [W]      first response token in each window
        response_len               [R]      response tokens per response
        labels                     [R, T]   per-response token labels, IGNORE-padded
        ids                        list[R]
    """

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        seqs, group, slot, starts = [], [], [], []
        for g, ex in enumerate(batch):
            for s, (ids, st) in enumerate(zip(ex["input_ids"], ex["response_start"])):
                seqs.append(ids)
                group.append(g)
                slot.append(s)
                starts.append(st)
        L = max(len(x) for x in seqs)
        T = max(len(ex["labels"]) for ex in batch)
        return {
            "input_ids": torch.tensor([x + [self.pad] * (L - len(x)) for x in seqs]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in seqs]),
            "group": torch.tensor(group),
            "slot": torch.tensor(slot),
            "response_start": torch.tensor(starts),
            "response_len": torch.tensor([len(ex["labels"]) for ex in batch]),
            "labels": torch.tensor([ex["labels"] + [IGNORE] * (T - len(ex["labels"])) for ex in batch]),
            "ids": [ex["id"] for ex in batch],
        }


class WindowBudgetSampler(Sampler[list[int]]):
    """Batches of whole responses with at most `max_windows` windows in total and, if
    `max_tokens` is set, at most `max_tokens` padded tokens (windows x longest window).
    A response over budget becomes a batch of its own.
    With `max_tokens`, responses of similar length are batched together to cut padding
    (shuffled: sorted within chunks of LENGTH_CHUNK responses, then batch order shuffled).
    Without it, batching is unchanged: plain shuffle, packed in order.
    Call set_epoch(e) each epoch to get a fresh, reproducible shuffle."""

    LENGTH_CHUNK = 1000

    def __init__(self, n_windows: list[int], max_windows: int = 32, shuffle: bool = False, seed: int = 42,
                 window_lens: list[list[int]] | None = None, max_tokens: int | None = None):
        if max_tokens and window_lens is None:
            raise ValueError("max_tokens needs window_lens")
        self.n_windows, self.max_windows = n_windows, max_windows
        self.window_lens, self.max_tokens = window_lens, max_tokens
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self._len = len(self._epoch_batches())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self, order: list[int]) -> list[list[int]]:
        batches, cur, used, longest = [], [], 0, 0
        for i in order:
            n = self.n_windows[i]
            ln = max(self.window_lens[i]) if self.max_tokens else 0
            if cur and (used + n > self.max_windows
                        or (self.max_tokens and (used + n) * max(longest, ln) > self.max_tokens)):
                batches.append(cur)
                cur, used, longest = [], 0, 0
            cur.append(i)
            used += n
            longest = max(longest, ln)
        if cur:
            batches.append(cur)
        return batches

    def _epoch_batches(self) -> list[list[int]]:
        order = list(range(len(self.n_windows)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        if not self.max_tokens:
            return self._batches(order)
        key = lambda i: max(self.window_lens[i])
        if self.shuffle:
            c = self.LENGTH_CHUNK
            order = [i for k in range(0, len(order), c) for i in sorted(order[k:k + c], key=key)]
        else:
            order.sort(key=key)
        batches = self._batches(order)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._epoch_batches()

    def __len__(self) -> int:
        # Packing depends on the shuffle, so this is exact only for shuffle=False.
        return self._len
