#!/usr/bin/env python3
# encoding: utf-8
"""
Loader that serves batches from a tensor already resident on the GPU.

RATIONALE
---------
The pipeline is CPU bound, not GPU bound. Measured on the five batch variants
of CNN_14L from the 2024 study (fit with error <4%):

    t_epoch = 8.09 ms x n_iterations + 7.70 s

The 8.09 ms per iteration are CPU overhead: Python loop, DataLoader collate,
`.to(device)` and above all TWO GPU synchronizations per iteration
(`loss.item()` and `.sum().item()`). That is why SimplePerceptron (7.8 K MACs)
and CNN_14L (425 M MACs) take almost the same time: 54,000x more compute,
1.8x more time.

THE RISK, AND WHY THIS CODE IS SO PEDANTIC
------------------------------------------
`00_train_models.py` uses `shuffle=True` in the training loader. That means
the RandomSampler **draws from the global CPU RNG in every epoch**. Any
replacement that does not replicate that consumption EXACTLY shifts the RNG
stream and changes the results -- silently, too.

What torch.utils.data.RandomSampler.__iter__ does with generator=None:

    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    yield from torch.randperm(n, generator=generator).tolist()

That is: ONE draw from the global RNG per epoch to seed a new generator, and
the randperm comes from that generator, not from the global one. It is
replicated to the letter below.

The BatchSampler groups those indices in order, without touching the RNG
again, so the batches are idx[0:bs], idx[bs:2bs], ...

VALIDATION
----------
It is not enough for the code to look correct: `validate_preload.py` trains N
seeds through the old path and through the new one and compares the hashes of
the final weights. If they are not identical, this is not used.
"""

import torch


def preload_dataset(dataset, device, batch_size=512):
    """
    Materializes the whole dataset as two tensors on `device`.

    The transforms (ToTensor + Normalize) are applied ONCE while iterating
    over the original dataset, exactly as the DataLoader would. This way the
    values are identical by construction, rather than by reimplementation.
    """
    from torch.utils.data import DataLoader

    # The preload uses DataLoaders, and each iteration consumes one draw from
    # the global RNG (`_base_seed`). If it were called after set_seed(), it
    # would shift the stream and the first trial would come out different --
    # which is exactly what happened in the validation before this was added.
    #
    # Saving and restoring the state makes the preload neutral whenever it is
    # called, instead of depending on the runner placing it in the right spot.
    rng_state = torch.random.get_rng_state()
    try:
        imgs, labs = [], []
        # num_workers=0 on purpose: the order is sequential and we do not want
        # the number of workers to influence anything.
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            imgs.append(x)
            labs.append(y)
        images = torch.cat(imgs).to(device)
        labels = torch.cat(labs).to(device)
    finally:
        torch.random.set_rng_state(rng_state)
    return images, labels


class PreloadedLoader:
    """
    DataLoader replacement for data already resident on the GPU.

    Reproduces the RNG consumption of RandomSampler so that the global stream
    stays exactly the same as with the original DataLoader.
    """

    def __init__(self, images, labels, batch_size, shuffle, drop_last=False):
        assert images.shape[0] == labels.shape[0]
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = images.shape[0]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def _indices(self):
        # A DataLoader draws from the global RNG EVERY TIME its iterator is
        # created, whether it shuffles or not: `_BaseDataLoaderIter.__init__`
        # draws `_base_seed` to seed the workers, even with num_workers=0.
        #
        # This matters a lot here: `train_model` iterates over the TEST loader
        # in every epoch (evaluate_model_with_loss), so the test loader also
        # shifts the stream. Ignoring it was the second reason why the first
        # version did not reproduce the results.
        if not self.shuffle:
            _base_seed = torch.empty((), dtype=torch.int64).random_().item()
            return torch.arange(self.n)

        # --- exact replica of the RNG consumption of a DataLoader(shuffle=True) ---
        #
        # Verified empirically (not deduced from the source code): per epoch
        # there are TWO draws from the global CPU RNG, in this order:
        #
        #   1. _BaseDataLoaderIter.__init__ draws `_base_seed`, which is used to
        #      seed the workers. It is drawn even with num_workers=0.
        #   2. RandomSampler.__iter__ draws the seed of the randperm generator.
        #
        # The order is NOT a cosmetic detail: swapping it consumes exactly the
        # same from the global RNG (the final state is identical) but produces
        # ANOTHER permutation, because it changes which of the two values seeds
        # the randperm. That is why comparing RNG states is not enough and the
        # indices have to be compared.
        _base_seed = torch.empty((), dtype=torch.int64).random_().item()   # (1)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())    # (2)
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.randperm(self.n, generator=g)

    def __iter__(self):
        idx = self._indices().to(self.images.device)
        last = self.n - (self.n % self.batch_size) if self.drop_last else self.n
        for i in range(0, last, self.batch_size):
            sel = idx[i:i + self.batch_size]
            yield self.images[sel], self.labels[sel]


def make_loaders(train_dataset, test_dataset, batch_size, device):
    """
    Equivalent to the two DataLoaders of 00_train_models.py:
        train: shuffle=True     test: shuffle=False
    """
    tr_x, tr_y = preload_dataset(train_dataset, device)
    te_x, te_y = preload_dataset(test_dataset, device)
    return (PreloadedLoader(tr_x, tr_y, batch_size, shuffle=True),
            PreloadedLoader(te_x, te_y, batch_size, shuffle=False))
