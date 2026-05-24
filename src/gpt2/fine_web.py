"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Streams and tokenizes the data and saves data shards to disk.
Run simply as:
$ python src/gpt2/fine_web.py
Will save the full sample to "data/edu_fineweb10B".
For a small smoke test:
$ python src/gpt2/fine_web.py --debug
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write a tiny 2M-token dataset to data/edu_fineweb_debug",
    )
    parser.add_argument("--remote-name", default="sample-10BT")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    args = parse_args()
    local_dir = "edu_fineweb_debug" if args.debug else "edu_fineweb10B"
    shard_size = args.shard_size or (int(1e6) if args.debug else int(1e8))
    max_shards = args.max_shards if args.max_shards is not None else (2 if args.debug else None)

    data_cache_dir = Path(args.data_root) / local_dir
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing shards to {data_cache_dir}")

    # stream the dataset so Hugging Face does not materialize a giant Arrow cache first
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.remote_name,
        split="train",
        streaming=True,
    )

    # tokenize all documents and write output shards, each of shard_size tokens
    cpu_count = os.cpu_count()
    assert cpu_count is not None
    nprocs = max(1, cpu_count // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        stopped_early = False
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = data_cache_dir / f"edufineweb_{split}_{shard_index:06d}"
                remainder = shard_size - token_count
                assert progress_bar is not None
                progress_bar.update(remainder)
                progress_bar.close()
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                if max_shards is not None and shard_index >= max_shards:
                    stopped_early = True
                    break
                progress_bar = None
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0 and not stopped_early:
            split = "val" if shard_index == 0 else "train"
            filename = data_cache_dir / f"edufineweb_{split}_{shard_index:06d}"
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
