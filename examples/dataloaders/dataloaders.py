import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
#from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm
import glob


def collate_batch(batch):
    batch = torch.stack([torch.as_tensor(b) for b in batch], 0)
    return batch, batch.clone()


def load_dataset(combine=50000, split="train", tokenizer=None, cache_path=None):
    token_chunks = []
    
    if cache_path is not None:
        # Pre-encoded
        with np.load(cache_path) as npz:
            for item in npz.files:
                token_chunks.append(npz[item])
            return token_chunks
        
    data = WikiText2(root="data", split=split)
    raw_text = ''
    
    print("Tokenizing dataset...")
    for line in tqdm(data):
        raw_text += line
    if len(raw_text) >= combine:
        tokens = np.stack(tokenizer.encode(raw_text))
        token_chunks.append(tokens)
        raw_text = ''
    else:
        raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(tokenizer.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def get_loader(batch_size, context_length=512, split="train", tokenizer=None, tok_name=None):
    data = lazy_load(tokenizer=tokenizer, split=split, tok_name=tok_name)[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length)
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_batch)

    return data_loader


def lazy_load(tokenizer, split="train", tok_name="none"):
    cache_path = 'cache_path_train_{}_{}.npz'.format(tok_name, split)
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(combine=1e99, split=split, tokenizer=tokenizer)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset(cache_path=cache_path)
    assert len(data) > 0
    return data
