
import os
import argparse
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.data_utils import (
    list_images, make_splits, save_splits, load_splits,
    print_split_summary, CLASS_NAMES,
)
from src.classical import build_feature_matrix, train_classical, load_classical_models, predict_classical
from src.cnn import build_model, get_dataloaders, train_cnn, predict_cnn, load_cnn_model, plot_training_history
from src.evaluate import (
    compute_metrics, save_metrics_csv,
    plot_confusion_matrix, analyze_errors, compare_methods,
)

PROJECT_DIR  = os.path.dirname(__file__)
SPLITS_PATH  = os.path.join(PROJECT_DIR, "results", "splits", "splits.json")
MODELS_DIR   = os.path.join(PROJECT_DIR, "results", "models")
CLASSICAL_MODELS = {
    "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
    "knn":    os.path.join(MODELS_DIR, "knn.pkl"),
    "svm":    os.path.join(MODELS_DIR, "svm.pkl"),
}
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn.pt")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Classificação de Screenshots de Jogos — Processamento de Imagens"
    )
    parser.add_argument(
        "--method", choices=["classical", "cnn", "both"], default="both",
        help="Método a executar (padrão: both)"
    )
    parser.add_argument(
        "--data-dir", default=PROJECT_DIR,
        help="Raiz do dataset com as 10 pastas de jogos"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=500,
        help="Imagens por classe no subconjunto (padrão: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente de aleatoriedade (padrão: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Épocas máximas de treino da CNN (padrão: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Tamanho do batch da CNN (padrão: 32)"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Re-treina mesmo que modelos já existam"
    )
    parser.add_argument(
        "--splits-path", default=SPLITS_PATH,
        help="Caminho do splits.json"
    )
    return parser.parse_args()



def _classical_models_exist() -> bool:
    return all(os.path.exists(p) for p in CLASSICAL_MODELS.values())


def _cnn_model_exists() -> bool:
    return os.path.exists(CNN_MODEL_PATH)


def _ensure_splits(args) -> tuple:
    if os.path.exists(args.splits_path) and not args.retrain:
        print(f"[main] Carregando splits existentes: {args.splits_path}")
        return load_splits(args.splits_path)

    print(f"[main] Criando splits (n_per_class={args.n_per_class}, seed={args.seed})...")
    df = list_images(args.data_dir, n_per_class=args.n_per_class, seed=args.seed)
    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print_split_summary(train_df, val_df, test_df)
    os.makedirs(os.path.dirname(args.splits_path), exist_ok=True)
    save_splits(train_df, val_df, test_df, args.splits_path)
    return train_df, val_df, test_df



def run_classical(train_df, test_df, retrain: bool) -> dict:
    print("\n" + "=" * 55)
    print("  PIPELINE CLÁSSICO")
    print("=" * 55)

    if _classical_models_exist() and not retrain:
        print("[main] Carregando modelos clássicos do disco...")
        models = load_classical_models(MODELS_DIR)
        print("[main] Extraindo features do teste...")
        X_test, y_test = build_feature_matrix(test_df, verbose=True)
    else:
        print("[main] Extraindo features de treino...")
        X_train, y_train = build_feature_matrix(train_df, verbose=True)
        print("[main] Extraindo features de teste...")
        X_test, y_test = build_feature_matrix(test_df, verbose=True)
        models = train_classical(X_train, y_train, models_dir=MODELS_DIR)

    preds = predict_classical(models, X_test)

    knn_metrics = compute_metrics(y_test, preds["knn_pred"], "k-NN (clássico)")
    knn_metrics["inference_time_ms"] = preds["knn_time_ms"]
    save_metrics_csv(knn_metrics, "knn")
    plot_confusion_matrix(y_test, preds["knn_pred"], title="Matriz de Confusão — k-NN")
    analyze_errors(y_test, preds["knn_pred"], method_name="k-NN")

    svm_metrics = compute_metrics(y_test, preds["svm_pred"], "SVM (clássico)")
    svm_metrics["inference_time_ms"] = preds["svm_time_ms"]
    save_metrics_csv(svm_metrics, "svm")
    plot_confusion_matrix(y_test, preds["svm_pred"], title="Matriz de Confusão — SVM")
    analyze_errors(y_test, preds["svm_pred"], method_name="SVM")

    np.save(os.path.join(MODELS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODELS_DIR, "y_test.npy"), y_test)

    return {"knn": knn_metrics, "svm": svm_metrics, "y_test": y_test}



def run_cnn(train_df, val_df, test_df, args) -> dict:
    print("\n" + "=" * 55)
    print("  CNN — MobileNetV2 (transfer learning)")
    print("=" * 55)

    loaders = get_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    if _cnn_model_exists() and not args.retrain:
        print("[main] Carregando melhor CNN do disco...")
        model = load_cnn_model(CNN_MODEL_PATH)
    else:
        model = build_model(num_classes=10)
        from src.cnn import count_trainable_params
        print(f"[main] Parâmetros treináveis: {count_trainable_params(model):,}")
        model, history = train_cnn(
            model, loaders,
            epochs=args.epochs,
            patience=5,
            save_path=CNN_MODEL_PATH,
        )
        plot_training_history(history)

    y_test_arr = np.concatenate([labels.numpy() for _, labels in loaders["test"]])
    y_pred_cnn, inf_time_ms = predict_cnn(model, loaders["test"])

    cnn_metrics = compute_metrics(y_test_arr, y_pred_cnn, "CNN (MobileNetV2)")
    cnn_metrics["inference_time_ms"] = inf_time_ms
    save_metrics_csv(cnn_metrics, "cnn")
    plot_confusion_matrix(y_test_arr, y_pred_cnn, title="Matriz de Confusão — CNN")
    analyze_errors(y_test_arr, y_pred_cnn, method_name="CNN")

    return {"cnn": cnn_metrics, "y_test": y_test_arr}



def main():
    args = parse_args()
    train_df, val_df, test_df = _ensure_splits(args)

    all_results = {}

    if args.method in ("classical", "both"):
        classical_out = run_classical(train_df, test_df, retrain=args.retrain)
        all_results["k-NN"] = classical_out["knn"]
        all_results["SVM"]  = classical_out["svm"]

    if args.method in ("cnn", "both"):
        cnn_out = run_cnn(train_df, val_df, test_df, args)
        all_results["CNN"] = cnn_out["cnn"]

    if args.method == "both" and len(all_results) == 3:
        compare_methods(all_results)

    print("\n[main] Concluído. Resultados em: results/")


if __name__ == "__main__":
    main()
