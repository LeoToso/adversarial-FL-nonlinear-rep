#!/usr/bin/env python3
"""Create the 10-letter, 150-client FEMNIST partition of Collins et al.

The source is an already-preprocessed LEAF FEMNIST corpus.  Train and test
files are pooled because Collins et al. repartition the underlying examples,
then a new 90/10 client-specific split is produced.  Sampling is without
replacement; the published generator accidentally reused examples, so this
script records the clean correction explicitly in metadata.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf_dir", default="data/femnist")
    parser.add_argument("--output_dir", default="data/femnist_collins")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clients", type=int, default=150)
    parser.add_argument("--classes_per_client", type=int, default=3)
    parser.add_argument("--max_per_class", type=int, default=4000)
    parser.add_argument("--target_mean_train", type=float, default=148.0)
    parser.add_argument("--min_train", type=int, default=50)
    parser.add_argument(
        "--published_compatibility",
        action="store_true",
        help=("Match the paper's reported sample statistics by cycling a "
              "class pool when it is exhausted. Reuse is recorded in metadata."),
    )
    return parser.parse_args()


def read_pool(leaf_dir: Path, max_per_class: int, rng: np.random.Generator):
    # Reservoir sampling avoids retaining every Python float from the complete
    # 817k-example LEAF corpus in memory.
    by_class = {label: [] for label in range(36, 46)}
    seen = {label: 0 for label in by_class}
    files = sorted(glob.glob(str(leaf_dir / "train" / "*.json")))
    files += sorted(glob.glob(str(leaf_dir / "test" / "*.json")))
    if not files:
        raise FileNotFoundError(f"No LEAF JSON files under {leaf_dir}")
    for path in files:
        with open(path) as stream:
            data = json.load(stream)
        for user_data in data["user_data"].values():
            for x, y in zip(user_data["x"], user_data["y"]):
                label = int(y)
                if label in by_class:
                    seen[label] += 1
                    item = np.asarray(x, dtype=np.float32)
                    if len(by_class[label]) < max_per_class:
                        by_class[label].append(item)
                    else:
                        replacement = int(rng.integers(seen[label]))
                        if replacement < max_per_class:
                            by_class[label][replacement] = item
    return {label: np.asarray(images, dtype=np.float32)
            for label, images in by_class.items()}


def concatenate(parts):
    offsets = [0]
    xs, ys = [], []
    for x, y in parts:
        xs.append(x); ys.append(y)
        offsets.append(offsets[-1] + len(y))
    return (np.concatenate(xs), np.concatenate(ys),
            np.asarray(offsets, dtype=np.int64))


def fit_disjoint_class_capacity(requested, n_clients, classes_per_client,
                                pool_sizes, min_train):
    """Reduce only allocations that exceed an observed class capacity.

    Each synthetic client receives an equal number of examples from each of
    its classes.  Decrementing one client's per-class allocation therefore
    reduces demand for all of that client's three labels without changing its
    class balance.  The paper's minimum of 50 training samples is preserved.
    """
    per_class = np.maximum(2, requested // classes_per_client).astype(int)
    min_total = int(np.ceil(min_train / 0.9))
    min_per_class = int(np.ceil(min_total / classes_per_client))

    assignments = [tuple((client + j) % 10
                         for j in range(classes_per_client))
                   for client in range(n_clients)]

    def demand(label):
        return sum(per_class[client] for client, labels in enumerate(assignments)
                   if label in labels)

    while True:
        excess = {
            label: demand(label) - pool_sizes[label]
            for label in range(10)
            if demand(label) > pool_sizes[label]
        }
        if not excess:
            break
        # Fix the most constrained class first.  Prefer reducing the largest
        # eligible client so the log-normal shape changes as little as possible.
        label = max(excess, key=excess.get)
        eligible = [client for client, labels in enumerate(assignments)
                    if label in labels and per_class[client] > min_per_class]
        if not eligible:
            raise RuntimeError(
                f"Class {label + 36} cannot satisfy the requested disjoint "
                f"partition while retaining at least {min_train} training "
                "samples per client."
            )
        client = max(eligible, key=lambda index: per_class[index])
        per_class[client] -= 1

    return per_class, assignments


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pool = read_pool(Path(args.leaf_dir), args.max_per_class, rng)
    for label in pool:
        rng.shuffle(pool[label])

    # Retain the log-normal allocation described in the paper while matching
    # its reported mean (148 training samples/client) and minimum (50).
    raw_counts = rng.lognormal(4.0, 1.0, args.n_clients)
    desired_train = (args.min_train +
                     (args.target_mean_train - args.min_train) *
                     raw_counts / raw_counts.mean())
    requested = np.ceil(desired_train / 0.9).astype(int)
    assignments = [tuple((client + j) % 10
                         for j in range(args.classes_per_client))
                   for client in range(args.n_clients)]
    if args.published_compatibility:
        per_class_counts = np.maximum(
            2, requested // args.classes_per_client).astype(int)
    else:
        per_class_counts, assignments = fit_disjoint_class_capacity(
            requested=requested,
            n_clients=args.n_clients,
            classes_per_client=args.classes_per_client,
            pool_sizes={label - 36: len(images)
                        for label, images in pool.items()},
            min_train=args.min_train,
        )
    cursors = {label: 0 for label in pool}
    assignments_per_original_class = {label: 0 for label in pool}
    train_parts, test_parts, class_sets = [], [], []
    for client in range(args.n_clients):
        labels = list(assignments[client])
        per_class = int(per_class_counts[client])
        client_x, client_y = [], []
        for relabelled in labels:
            original = relabelled + 36
            start, stop = cursors[original], cursors[original] + per_class
            assignments_per_original_class[original] += per_class
            if stop > len(pool[original]) and not args.published_compatibility:
                raise RuntimeError(
                    f"Class {original} has {len(pool[original])} examples but "
                    f"the requested disjoint partition needs at least {stop}. "
                    "Increase --max_per_class only if the LEAF pool contains more."
                )
            if stop <= len(pool[original]):
                selected = pool[original][start:stop]
            else:
                # Compatibility mode uses a deterministic circular pool.  This
                # matches the published sample counts with the finite lowercase
                # class corpus while making reuse explicit and auditable.
                indices = np.arange(start, stop) % len(pool[original])
                selected = pool[original][indices]
            client_x.append(selected)
            client_y.append(np.full(per_class, relabelled, dtype=np.int64))
            cursors[original] = (stop % len(pool[original])
                                 if args.published_compatibility else stop)
        x, y = np.concatenate(client_x), np.concatenate(client_y)
        order = rng.permutation(len(y)); x, y = x[order], y[order]
        train_len = int(0.9 * len(y))
        train_parts.append((x[:train_len], y[:train_len]))
        test_parts.append((x[train_len:], y[train_len:]))
        class_sets.append(labels)

    train_x, train_y, train_offsets = concatenate(train_parts)
    test_x, test_y, test_offsets = concatenate(test_parts)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "partition.npz", train_x=train_x, train_y=train_y,
        train_offsets=train_offsets, test_x=test_x, test_y=test_y,
        test_offsets=test_offsets,
    )
    reused_by_class = {
        str(label): max(0, assignments_per_original_class[label] - len(pool[label]))
        for label in pool
    }
    reused_total = int(sum(reused_by_class.values()))
    metadata = {
        "benchmark": "collins21_femnist_letters",
        "n_clients": args.n_clients,
        "n_classes": 10,
        "original_labels": list(range(36, 46)),
        "classes_per_client": args.classes_per_client,
        "class_sets": class_sets,
        "seed": args.seed,
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "mean_train_samples_per_client": float(np.mean(np.diff(train_offsets))),
        "min_train_samples_per_client": int(np.min(np.diff(train_offsets))),
        "target_mean_train_samples_per_client": args.target_mean_train,
        "target_min_train_samples_per_client": args.min_train,
        "split": "90/10 per synthetic client",
        "published_compatibility": args.published_compatibility,
        "sample_reuse": bool(reused_total),
        "reused_assignments": reused_total,
        "reused_assignments_by_original_class": reused_by_class,
        "reuse_fraction": float(reused_total / (len(train_y) + len(test_y))),
        "note": (
            "Published-compatible sample statistics with explicitly recorded "
            "class-pool cycling" if args.published_compatibility else
            "Clean no-overlap implementation of Collins et al. 10-letter protocol"
        ),
    }
    with (output / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
