#!/usr/bin/env python3
# encoding: utf-8

from collections import Counter
from torchvision import datasets, transforms

from utils.log_utils import log, logTable

train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

train_counts = Counter(train_dataset.targets.numpy())
log("Images in train dataset by class:")
accum_count = 0
for label, count in sorted(train_counts.items()):
    log(f"Class {label}: with {count} images")
    accum_count += count

log(f"Total num imag {accum_count}")

# Count the images per class in the test set
test_counts = Counter(test_dataset.targets.numpy())
log("\nImages in test dataset by class:")
accum_count = 0
for label, count in sorted(test_counts.items()):
    log(f"Class {label}: with {count} images")
    accum_count += count

log(f"Total num imag {accum_count}")
