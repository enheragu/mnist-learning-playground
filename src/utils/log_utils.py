
#!/usr/bin/env python3
# encoding: utf-8

from tqdm import tqdm

## log in terminal without affecting tqdm bar
def log(*args, **kwargs):
    tqdm.write(*args, **kwargs)   