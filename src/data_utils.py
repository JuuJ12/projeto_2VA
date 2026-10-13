
import os
import json
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CLASS_NAMES = [
    "Among Us",
    "Apex Legends",
    "Fortnite",
    "Forza Horizon",
    "Free Fire",
    "Genshin Impact",
    "God of War",
    "Minecraft",
    "Roblox",
    "Terraria",
]


def list_images(
    data_dir: str,
    n_per_class: int = 500,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    if class_names is None:
        class_names = CLASS_NAMES

    rng = random.Random(seed)
    rows = []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Pasta da classe '{cls}' não encontrada em: {cls_dir}"
            )

        all_files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if not all_files:
            raise ValueError(f"Nenhuma imagem encontrada em: {cls_dir}")

        if n_per_class is not None and n_per_class < len(all_files):
            sampled = rng.sample(all_files, n_per_class)
        else:
            sampled = all_files

        for path in sampled:
            rows.append({"path": path, "label": cls, "label_idx": idx})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio deve ser menor que 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=seed,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        stratify=temp_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: str,
) -> None:
    splits = {
        "train": train_df[["path", "label", "label_idx"]].to_dict(orient="records"),
        "val": val_df[["path", "label", "label_idx"]].to_dict(orient="records"),
        "test": test_df[["path", "label", "label_idx"]].to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"[data_utils] Splits salvos em: {out_path}")


def load_splits(splits_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_df = pd.DataFrame(splits["train"])
    val_df = pd.DataFrame(splits["val"])
    test_df = pd.DataFrame(splits["test"])
    return train_df, val_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n{'=' * 45}")
    print(f"{'RESUMO DOS SPLITS':^45}")
    print(f"{'=' * 45}")
    print(f"  Treino:    {len(train_df):>5} imagens  ({len(train_df)/total*100:.1f}%)")
    print(f"  Validação: {len(val_df):>5} imagens  ({len(val_df)/total*100:.1f}%)")
    print(f"  Teste:     {len(test_df):>5} imagens  ({len(test_df)/total*100:.1f}%)")
    print(f"  Total:     {total:>5} imagens")
    print(f"{'=' * 45}")

    print("\n  Imagens por classe (treino | val | teste):")
    for cls in CLASS_NAMES:
        n_tr = (train_df["label"] == cls).sum()
        n_va = (val_df["label"] == cls).sum()
        n_te = (test_df["label"] == cls).sum()
        print(f"    {cls:<18} {n_tr:>4} | {n_va:>3} | {n_te:>3}")
    print()
