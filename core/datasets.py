"""
datasets.py  —  Federated data loaders
========================================
Supports CIFAR-10, CIFAR-100, FEMNIST (EMNIST-Letters proxy),
and Sent140 (LEAF JSON or AG-News proxy).

Heterogeneity is controlled by a Dirichlet alpha:
  alpha → 0   extreme non-IID (each client ~ 1 class)
  alpha → ∞   IID

Each load_* function returns:
  client_train_loaders : List[DataLoader]  — local training data per client
  client_test_loaders  : List[DataLoader]  — local test data per client (80/20 split)
  global_test_loader   : DataLoader        — global test set
"""

import os, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple


# ── Dirichlet partitioner ────────────────────────────────────────────────────

def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_per_client: int = 8,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    class_idx = [np.where(labels == c)[0].copy() for c in range(n_classes)]
    for ci in class_idx:
        rng.shuffle(ci)

    client_idx: List[List[int]] = [[] for _ in range(n_clients)]
    for ci in class_idx:
        if len(ci) == 0:
            continue
        props  = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (props * len(ci)).astype(int)
        counts[-1] = len(ci) - counts[:-1].sum()
        counts = np.clip(counts, 0, None)
        ptr = 0
        for k, cnt in enumerate(counts):
            client_idx[k].extend(ci[ptr: ptr + cnt].tolist())
            ptr += cnt

    # guarantee minimum samples
    for k in range(n_clients):
        if len(client_idx[k]) < min_per_client:
            donor = max(range(n_clients), key=lambda i: len(client_idx[i]))
            take  = client_idx[donor][:min_per_client]
            client_idx[donor] = client_idx[donor][min_per_client:]
            client_idx[k].extend(take)
    return client_idx



def load_heart_disease(n_clients, alpha, data_dir="./data", batch_size=32, seed=42):
    """
    Fed-Heart-Disease: 4 natural client splits (Cleveland, Hungary, Switzerland, VA).
    Binary classification: 0 = no disease, 1 = disease (original labels 1-4).
    n_clients and alpha are ignored — natural splits are used.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    files = {
        0: "processed.cleveland.data",
        1: "processed.hungarian.data",
        2: "processed.switzerland.data",
        3: "processed.va.data",
    }
    heart_dir = os.path.join(data_dir, "heart_disease")

    col_names = ["age","sex","cp","trestbps","chol","fbs","restecg",
                 "thalach","exang","oldpeak","slope","ca","thal","target"]

    # load all clients first without dropping columns
    raw_dfs = []
    for c, fname in files.items():
        df = pd.read_csv(os.path.join(heart_dir, fname),
                         header=None, names=col_names, na_values="?")
        df["target"] = (df["target"] > 0).astype(int)
        raw_dfs.append(df)

    # concatenate globally, then drop columns with >50% missing across all data
    full_df = pd.concat(raw_dfs, ignore_index=True)
    thresh = int(0.5 * len(full_df))
    full_df = full_df.dropna(axis=1, thresh=thresh)
    full_df = full_df.fillna(full_df.median())
    full_df = full_df.reset_index(drop=True)

    # rebuild client indices based on original sizes
    client_indices = []
    offset = 0
    for df in raw_dfs:
        client_indices.append(list(range(offset, offset + len(df))))
        offset += len(df)

    X = full_df.drop("target", axis=1).values.astype(np.float32)
    y = full_df["target"].values.astype(np.int64)

    # normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    class TabularDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    full_ds = TabularDataset(X, y)

    client_train_loaders, client_test_loaders = [], []
    all_test_indices = []
    for idx in client_indices:
        train_idx, test_idx = train_test_split_indices(idx, test_ratio=0.1, seed=seed)
        client_train_loaders.append(DataLoader(Subset(full_ds, train_idx),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0))
        client_test_loaders.append(DataLoader(Subset(full_ds, test_idx),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0))
        all_test_indices.extend(test_idx)

    global_test_loader = DataLoader(Subset(full_ds, all_test_indices),
                                    batch_size=256, shuffle=False, num_workers=0)
    return client_train_loaders, client_test_loaders, global_test_loader




def train_test_split_indices(indices, test_ratio=0.2, seed=42):
    """Split a list of indices into train and test subsets."""
    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_ratio))
    return indices[n_test:].tolist(), indices[:n_test].tolist()


# ── CIFAR-10 ─────────────────────────────────────────────────────────────────

def load_cifar10(n_clients, alpha, data_dir="./data", batch_size=64, seed=42):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tr)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=te)
    splits   = dirichlet_partition(np.array(train_ds.targets), n_clients, alpha, seed)
    client_train_loaders, client_test_loaders = [], []
    for idx in splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(train_ds, train_idx),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=False))
        client_test_loaders.append(DataLoader(Subset(train_ds, test_idx),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=False))
    return client_train_loaders, client_test_loaders, \
           DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)


# ── CIFAR-100 ────────────────────────────────────────────────────────────────

def load_cifar100(n_clients, alpha, data_dir="./data", batch_size=64, seed=42):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR100(data_dir, train=True,  download=True, transform=tr)
    test_ds  = datasets.CIFAR100(data_dir, train=False, download=True, transform=te)
    splits   = dirichlet_partition(np.array(train_ds.targets), n_clients, alpha, seed)
    client_train_loaders, client_test_loaders = [], []
    for idx in splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(train_ds, train_idx),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=False))
        client_test_loaders.append(DataLoader(Subset(train_ds, test_idx),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=False))
    return client_train_loaders, client_test_loaders, \
           DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)


# ── FEMNIST ───────────────────────────────────────────────────────────────────

class RelabelDataset(Dataset):
    def __init__(self, dataset, offset=-1):
        self.dataset = dataset
        self.offset  = offset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, y + self.offset

def load_femnist_leaf(data_dir="./data/femnist", batch_size=32, seed=42):
    """Load FEMNIST using natural per-writer heterogeneity from LEAF benchmark.
    Download LEAF first: https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
    Expected structure: data_dir/train/*.json, data_dir/test/*.json
    """
    import glob, json

    def _read(split):
        all_x, all_y, client_splits = [], [], []
        fps = sorted(glob.glob(os.path.join(data_dir, split, "*.json")))
        if not fps:
            raise FileNotFoundError(f"No LEAF FEMNIST JSON files found in {data_dir}/{split}/")
        for fp in fps:
            d = json.load(open(fp))
            for user_data in d["user_data"].values():
                idx_start = len(all_x)
                x = np.array(user_data["x"], dtype=np.float32).reshape(-1, 1, 28, 28)
                y = np.array(user_data["y"], dtype=np.int64)
                all_x.append(x)
                all_y.append(y)
                client_splits.append(list(range(idx_start, idx_start + len(y))))
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        return all_x, all_y, client_splits

    tr_x, tr_y, client_splits = _read("train")
    te_x, te_y, _             = _read("test")

    # normalize
    mean, std = 0.1307, 0.3081
    tr_x = (tr_x - mean) / std
    te_x = (te_x - mean) / std

    class NumpyDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x)
            self.y = torch.tensor(y)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], self.y[i]

    train_ds = NumpyDataset(tr_x, tr_y)
    test_ds  = NumpyDataset(te_x, te_y)

    client_train_loaders, client_test_loaders = [], []
    for idx in client_splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(train_ds, train_idx),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0))
        client_test_loaders.append(DataLoader(Subset(train_ds, test_idx),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0))
    return client_train_loaders, client_test_loaders, \
           DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)


def load_femnist(n_clients, alpha, data_dir="./data", batch_size=32, seed=42, use_leaf=False):
    if use_leaf:
        leaf_dir = os.path.join(data_dir, "femnist")
        return load_femnist_leaf(data_dir=leaf_dir, batch_size=batch_size, seed=seed)

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_ds = datasets.EMNIST(data_dir, split="letters", train=True,  download=True, transform=tr)
    test_ds  = datasets.EMNIST(data_dir, split="letters", train=False, download=True, transform=tr)
    train_ds = RelabelDataset(train_ds, offset=-1)
    test_ds  = RelabelDataset(test_ds,  offset=-1)
    labels   = np.array(train_ds.dataset.targets) - 1
    splits   = dirichlet_partition(labels, n_clients, alpha, seed)
    client_train_loaders, client_test_loaders = [], []
    for idx in splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(train_ds, train_idx),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0))
        client_test_loaders.append(DataLoader(Subset(train_ds, test_idx),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0))
    return client_train_loaders, client_test_loaders, \
           DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)


# ── Sent140 ───────────────────────────────────────────────────────────────────

class _VecDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _tfidf(tr_texts, te_texts, max_f=2000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=max_f, sublinear_tf=True)
    return v.fit_transform(tr_texts).toarray().astype(np.float32), \
           v.transform(te_texts).toarray().astype(np.float32)


def load_sent140(n_clients, alpha, data_dir="./data", batch_size=32, seed=42):
    leaf = os.path.join(data_dir, "sent140")
    if os.path.isdir(leaf) and glob.glob(os.path.join(leaf, "train", "*.json")):
        return _from_leaf(leaf, n_clients, alpha, batch_size, seed)
    print("[INFO] Sent140 LEAF not found — using AG-News 2-class proxy.")
    return _agnews_proxy(n_clients, alpha, data_dir, batch_size, seed)


def load_isic2019(n_clients, alpha, data_dir="./data", batch_size=32, seed=42):
    """
    Fed-ISIC2019: 6 natural client splits from FLamby train_test_split file.
    9-class skin lesion classification.
    n_clients and alpha are ignored — natural splits are used.
    """
    import pandas as pd
    from torchvision import transforms

    split_file = os.path.join(
        os.path.expanduser("~/adv_fedrep/data/FLamby/flamby/datasets/fed_isic2019"),
        "dataset_creation_scripts/train_test_split"
    )
    img_dir = os.path.join(data_dir, "isic2019/ISIC_2019_Training_Input_preprocessed")

    df = pd.read_csv(split_file)

    train_tf = transforms.Compose([
        transforms.RandomCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ISICDataset(Dataset):
        def __init__(self, images, targets, transform):
            self.images = images
            self.targets = targets
            self.transform = transform
        def __len__(self): return len(self.images)
        def __getitem__(self, i):
            from PIL import Image as PILImage
            img = PILImage.open(self.images[i]).convert("RGB")
            img = self.transform(img)
            return img, torch.tensor(self.targets[i], dtype=torch.long)

    client_train_loaders, client_test_loaders = [], []
    all_test_images, all_test_targets = [], []

    for center in range(6):
        train_df = df[df["fold2"] == f"train_{center}"].reset_index(drop=True)
        test_df  = df[df["fold2"] == f"test_{center}"].reset_index(drop=True)

        train_imgs = [os.path.join(img_dir, f"{img}.jpg") for img in train_df["image"]]
        test_imgs  = [os.path.join(img_dir, f"{img}.jpg") for img in test_df["image"]]

        train_ds = ISICDataset(train_imgs, train_df["target"].tolist(), train_tf)
        test_ds  = ISICDataset(test_imgs,  test_df["target"].tolist(),  test_tf)

        client_train_loaders.append(DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, num_workers=4,
                                               pin_memory=True))
        client_test_loaders.append(DataLoader(test_ds, batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              pin_memory=True))
        all_test_images.extend(test_imgs)
        all_test_targets.extend(test_df["target"].tolist())

    global_test_ds = ISICDataset(all_test_images, all_test_targets, test_tf)
    global_test_loader = DataLoader(global_test_ds, batch_size=256, shuffle=False,
                                    num_workers=4, pin_memory=True)

    return client_train_loaders, client_test_loaders, global_test_loader


def _from_leaf(leaf, n_clients, alpha, batch_size, seed):
    def _read(split):
        texts, labels = [], []
        for fp in glob.glob(os.path.join(leaf, split, "*.json")):
            d = json.load(open(fp))
            for ud in d["user_data"].values():
                for x, y in zip(ud["x"], ud["y"]):
                    texts.append(x[4]); labels.append(int(int(y) > 0))
        return texts, labels
    tr_t, tr_l = _read("train"); te_t, te_l = _read("test")
    X_tr, X_te = _tfidf(tr_t, te_t)
    splits = dirichlet_partition(np.array(tr_l), n_clients, alpha, seed)
    tr_ds = _VecDS(X_tr, tr_l)
    client_train_loaders, client_test_loaders = [], []
    for idx in splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(tr_ds, train_idx),
                                               batch_size=batch_size, shuffle=True))
        client_test_loaders.append(DataLoader(Subset(tr_ds, test_idx),
                                              batch_size=batch_size, shuffle=False))
    return client_train_loaders, client_test_loaders, \
           DataLoader(_VecDS(X_te, te_l), batch_size=256, shuffle=False)


def _agnews_proxy(n_clients, alpha, data_dir, batch_size, seed):
    try:
        from torchtext.datasets import AG_NEWS
        def _parse(split):
            rows = [(l-1, t) for l, t in AG_NEWS(root=data_dir, split=split) if l in (1,2)]
            return [l for l,_ in rows], [t for _,t in rows]
        tr_l, tr_t = _parse("train"); te_l, te_t = _parse("test")
        X_tr, X_te = _tfidf(tr_t, te_t)
    except Exception as e:
        print(f"[WARN] AG-News failed ({e}). Using synthetic data.")
        rng = np.random.default_rng(seed)
        X_tr = rng.standard_normal((8000, 2000)).astype(np.float32)
        tr_l = rng.integers(0, 2, 8000).tolist()
        X_te = rng.standard_normal((2000, 2000)).astype(np.float32)
        te_l = rng.integers(0, 2, 2000).tolist()
    splits = dirichlet_partition(np.array(tr_l), n_clients, alpha, seed)
    tr_ds  = _VecDS(X_tr, tr_l)
    client_train_loaders, client_test_loaders = [], []
    for idx in splits:
        train_idx, test_idx = train_test_split_indices(idx, seed=seed)
        client_train_loaders.append(DataLoader(Subset(tr_ds, train_idx),
                                               batch_size=batch_size, shuffle=True))
        client_test_loaders.append(DataLoader(Subset(tr_ds, test_idx),
                                              batch_size=batch_size, shuffle=False))
    return client_train_loaders, client_test_loaders, \
           DataLoader(_VecDS(X_te, te_l), batch_size=256, shuffle=False)


# ── Registry ──────────────────────────────────────────────────────────────────

DATASET_META: Dict[str, Dict] = {
    "cifar10":  {"n_classes": 10,  "in_ch": 3, "img": 32, "dim": 3*32*32},
    "cifar100": {"n_classes": 100, "in_ch": 3, "img": 32, "dim": 3*32*32},
    "femnist":  {"n_classes": 26,  "in_ch": 1, "img": 28, "dim": 1*28*28},
    "sent140":  {"n_classes": 2,   "in_ch": 1, "img": 0,  "dim": 2000},
    "heart_disease": {"n_classes": 2, "in_ch": 1, "img": 0, "dim": 11},
    "isic2019": {"n_classes": 9, "in_ch": 3, "img": 200, "dim": 3*200*200},
}

_LOADERS = {"cifar10": load_cifar10, "cifar100": load_cifar100,
            "femnist": load_femnist, "sent140": load_sent140, "heart_disease": load_heart_disease, "isic2019": load_isic2019}


def get_loaders(dataset, n_clients, alpha, data_dir="./data", batch_size=64, seed=42, use_leaf=False):
    if dataset not in _LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {list(_LOADERS)}")
    if dataset == "femnist":
        return load_femnist(n_clients, alpha, data_dir, batch_size, seed, use_leaf=use_leaf)
    return _LOADERS[dataset](n_clients, alpha, data_dir, batch_size, seed)


def heterogeneity_score(client_loaders, n_classes) -> float:
    """Mean pairwise L1 distance of per-client label frequency vectors."""
    freqs = []
    for ld in client_loaders:
        cnt = np.zeros(n_classes)
        for _, y in ld:
            for lbl in y.numpy(): cnt[int(lbl)] += 1
        freqs.append(cnt / (cnt.sum() + 1e-9))
    freqs = np.array(freqs); n = len(freqs)
    s = sum(np.abs(freqs[i] - freqs[j]).sum()
            for i in range(n) for j in range(i+1,n))
    return float(s / max(n*(n-1)/2, 1))