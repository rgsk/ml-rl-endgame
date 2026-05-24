import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="read stats from data/edu_fineweb_debug instead of data/edu_fineweb10B",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path("data/edu_fineweb_debug" if args.debug else "data/edu_fineweb10B")
    shards = sorted(data_dir.glob("*.npy"))

    print(f"found {len(shards)} shards in {data_dir}")
    for shard in shards:
        tokens = np.load(shard, mmap_mode="r")
        print(
            f"{shard.name}: shape={tokens.shape}, dtype={tokens.dtype}, tokens={tokens.size:,}"
        )


if __name__ == "__main__":
    main()
