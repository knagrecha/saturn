# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torchtext.datasets import WikiText2, WikiText103
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm


def collate_batch(batch):
    batch = torch.stack([torch.as_tensor(b) for b in batch], 0)
    return batch, batch.clone()


def load_dataset(combine=50000, split="train", tokenizer=None, cache_path=None, full_data=False):
    token_chunks = []
    
    if cache_path is not None:
        # Pre-encoded
        with np.load(cache_path) as npz:
            for item in npz.files:
                token_chunks.append(npz[item])
            return token_chunks
    
    if not full_data:
        data = WikiText2(root="data", split=split)
    else:
        data = WikiText103(root="data", split=split)
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


def get_loader(batch_size, context_length=512, split="train", tokenizer=None, tok_name=None, full_data=False):
    data = lazy_load(tokenizer=tokenizer, split=split, tok_name=tok_name, full_data=full_data)[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length)
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_batch)

    return data_loader


def lazy_load(tokenizer, split="train", tok_name="none", full_data=False):
    if not full_data:
        cache_path = 'cache_path_train_{}_{}.npz'.format(tok_name, split)
    else:
        cache_path = 'cache_path_train_{}_{}_103.npz'.format(tok_name, split)
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(combine=1e99, split=split, tokenizer=tokenizer, full_data=full_data)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset(cache_path=cache_path)
    assert len(data) > 0
    return data
