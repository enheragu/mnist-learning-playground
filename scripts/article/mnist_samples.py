"""Sample digits of the methodology of the article: the first image of each class in the MNIST training set.

Usage: python3 mnist_samples.py <out_dir> [data_dir]
Needs the MNIST dataset instead of the deposit; it downloads it to data_dir (./data by default) when it is missing.
"""
import os
import sys

from torchvision import datasets

out_dir = sys.argv[1]
data_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
os.makedirs(out_dir, exist_ok=True)

dataset = datasets.MNIST(data_dir, train=True, download=True)
seen = set()
for index, (image, label) in enumerate(dataset):
    if label not in seen:
        seen.add(label)
        image.save(os.path.join(out_dir, f"image_{index}.png"))
        print(f"image_{index}.png: digit {label}")
    if len(seen) == 10:
        break
