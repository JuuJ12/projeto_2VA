"""
evaluate.py
===========
Métricas, visualizações e comparação entre pipeline clássico e CNN.

Protocolo de comparação (acordado na entrevista):
    - Mesmo conjunto de teste (750 imagens) para ambos os métodos
    - Métricas: acurácia top-1, F1-macro, relatório por classe
    - Tempo de inferência (extração + classificação, sem treino)
    - Matriz de confusão 10×10
    - Análise de erros: 3 pares de classes mais confundidos por método

Decisões registradas:
    GLOSSARY — F1-macro, Tempo de inferência, Conjunto de teste
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display para salvar figuras
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data_utils import CLASS_NAMES

# Diretório padrão para figuras
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "figures"
)
METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "metrics"
)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str = "método",
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """
    Calcula acurácia, F1-macro e relatório completo por classe.

    Parâmetros
    ----------
    y_true, y_pred : arrays de rótulos inteiros.
    method_name    : nome do método (para logging e relatório).
    class_names    : lista com os nomes canônicos das classes.

    Retorna
    -------
    dict com chaves:
        "accuracy"   – float
        "f1_macro"   – float
        "report"     – str (classification_report completo)
        "report_df"  – pd.DataFrame por classe
    """
    acc     = accuracy_score(y_true, y_pred)
    f1_mac  = f1_score(y_true, y_pred, average="macro")
    report  = classification_report(y_true, y_pred, target_names=class_names)
    rep_df  = pd.DataFrame(
        classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    ).T

    print(f"\n{'=' * 55}")
    print(f"  {method_name}")
    print(f"{'=' * 55}")
    print(f"  Acurácia : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-macro : {f1_mac:.4f}")
    print(f"\n{report}")

    return {
        "accuracy":  acc,
        "f1_macro":  f1_mac,
        "report":    report,
        "report_df": rep_df,
    }


def save_metrics_csv(
    metrics: dict,
    method_name: str,
    out_dir: str = METRICS_DIR,
) -> None:
    """Salva o relatório por classe em CSV."""
    os.makedirs(out_dir, exist_ok=True)
    slug = method_name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"{slug}_metrics.csv")
    metrics["report_df"].to_csv(path)
    print(f"[evaluate] Métricas salvas em: {path}")


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão",
    class_names: list[str] = CLASS_NAMES,
    out_path: str | None = None,
) -> None:
    """
    Plota e salva a matriz de confusão normalizada por linha (recall).

    Parâmetros
    ----------
    y_true, y_pred : arrays de rótulos inteiros.
    title          : título da figura.
    class_names    : rótulos dos eixos.
    out_path       : caminho para salvar (PNG). Se None, usa FIGURES_DIR.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if out_path is None:
        slug = title.lower().replace(" ", "_").replace("/", "_")
        out_path = os.path.join(FIGURES_DIR, f"confusion_{slug}.png")

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normaliza por linha

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Classe Predita",
        ylabel="Classe Verdadeira",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anota células com o valor
    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Matriz de confusão salva em: {out_path}")


# ---------------------------------------------------------------------------
# Análise de erros
# ---------------------------------------------------------------------------

def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str = "método",
    top_k: int = 3,
    class_names: list[str] = CLASS_NAMES,
) -> list[dict]:
    """
    Identifica os `top_k` pares de classes mais confundidas.

    Retorna
    -------
    Lista de dicts com chaves:
        "true_class", "pred_class", "count", "pct_of_true"
    Ordenada de mais para menos confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # remove acertos — só confusões

    # Achata, ordena e pega os top_k
    flat_idx = np.argsort(cm.flatten())[::-1][:top_k]
    errors = []
    for idx in flat_idx:
        i, j = divmod(idx, len(class_names))
        total_true = (y_true == i).sum()
        pct = cm[i, j] / total_true if total_true > 0 else 0
        errors.append({
            "true_class": class_names[i],
            "pred_class": class_names[j],
            "count": int(cm[i, j]),
            "pct_of_true": float(pct),
        })

    print(f"\n[{method_name}] Top-{top_k} confusões:")
    for e in errors:
        print(
            f"  {e['true_class']:>18} → {e['pred_class']:<18}"
            f"  {e['count']:>3} casos ({e['pct_of_true']*100:.1f}% da classe)"
        )
    return errors


# ---------------------------------------------------------------------------
# Comparação lado a lado
# ---------------------------------------------------------------------------

def compare_methods(
    results: dict,
    out_dir: str = FIGURES_DIR,
) -> None:
    """
    Gera tabela resumo e gráfico de barras comparando todos os métodos.

    Parâmetros
    ----------
    results : dict mapeando nome_do_método → dict com chaves
              "accuracy", "f1_macro" e opcionalmente "inference_time_ms".
    out_dir : diretório para salvar a figura.
    """
    os.makedirs(out_dir, exist_ok=True)

    methods = list(results.keys())
    accs    = [results[m]["accuracy"]  for m in methods]
    f1s     = [results[m]["f1_macro"]  for m in methods]
    times   = [results[m].get("inference_time_ms", 0) for m in methods]

    # Tabela no terminal
    print(f"\n{'=' * 60}")
    print(f"{'COMPARAÇÃO DOS MÉTODOS':^60}")
    print(f"{'=' * 60}")
    print(f"{'Método':<20} {'Acurácia':>10} {'F1-macro':>10} {'Tempo(ms)':>12}")
    print(f"{'-' * 60}")
    for m, acc, f1, t in zip(methods, accs, f1s, times):
        print(f"{m:<20} {acc:>10.4f} {f1:>10.4f} {t:>12.1f}")
    print(f"{'=' * 60}")

    # Gráfico de barras
    x   = np.arange(len(methods))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Acurácia", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, f1s,  w, label="F1-macro", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Comparação: Pipeline Clássico vs CNN")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "comparison_methods.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Gráfico de comparação salvo em: {out_path}")
