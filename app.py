"""
app.py
======
Interface Interativa em Streamlit para Classificação de Jogos Eletrônicos.
Comparação detalhada entre Processamento Clássico de Imagens e Redes Neurais Convolucionais (CNN).
"""

import os
import sys
import time
import json
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Garante que os módulos locais em 'src' sejam importados
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import CLASS_NAMES, load_splits
from src.preprocessing import preprocess, apply_gaussian
from src.features import extract_features, feature_names
from src.classical import load_classical_models
from src.cnn import load_cnn_model, EVAL_TRANSFORM
import torch


# ---------------------------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Classificação de Jogos — Processamento de Imagens",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS personalizada
# OBS: as classes .card-* vêm ANTES das .badge-* de propósito. Como têm a mesma
# especificidade, a que vem depois ganha, então os badges mantêm as cores deles.
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #1E293B;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* ---------- Cards claros (texto escuro garantido no tema escuro) ---------- */
    .card-claro {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 8px;
        color: #1E293B !important;
    }
    .card-claro * {
        color: #1E293B;
    }
    .card-claro code {
        color: #0F766E !important;
        background: #E2E8F0;
    }
    .card-diagnostico {
        margin-top: 15px;
        padding: 12px 18px;
        border-radius: 8px;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E293B !important;
    }
    .card-diagnostico * {
        color: #1E293B;
    }
    .card-diagnostico code {
        color: #0F766E !important;
        background: #DBEAFE;
    }
    .card-conclusao {
        background-color: #F8FAFC;
        border-left: 4px solid #16A34A;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 15px;
        color: #1E293B !important;
    }
    .card-conclusao * {
        color: #1E293B;
    }

    /* ---------- Badges (precisam vir DEPOIS dos .card-* ) ---------- */
    .badge-correct {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-wrong {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-pred {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* ---------- Passo a passo ---------- */
    .step-box {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #1E293B;
    }
    .step-box h4 {
        color: #1E3A8A;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Funções de Cache de Recursos e Dados
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models():
    """Carrega os modelos clássicos e a CNN em cache."""
    models_dir = os.path.join(PROJECT_ROOT, "results", "models")
    classical_models = load_classical_models(models_dir)
    cnn_model = load_cnn_model(os.path.join(models_dir, "best_cnn.pt"))
    return classical_models, cnn_model


@st.cache_data
def get_splits_data():
    """Carrega as divisões do dataset garantindo resolução correta dos caminhos."""
    splits_path = os.path.join(PROJECT_ROOT, "results", "splits", "splits.json")
    if os.path.exists(splits_path):
        train_df, val_df, test_df = load_splits(splits_path)
        return train_df, val_df, test_df
    return None, None, None


@st.cache_data
def get_metrics_tables():
    """Carrega os arquivos CSV de métricas por classe."""
    metrics = {}
    for name in ["knn", "svm", "cnn"]:
        csv_path = os.path.join(PROJECT_ROOT, "results", "metrics", f"{name}_metrics.csv")
        if os.path.exists(csv_path):
            metrics[name] = pd.read_csv(csv_path, index_col=0)
    return metrics


# ---------------------------------------------------------------------------
# Funções de Processamento e Visualização de Imagem Única
# ---------------------------------------------------------------------------
def compute_visual_transformations(img_bgr: np.ndarray):
    """
    Computa as visualizações intermediárias do pipeline para inspeção visual:
    1. Imagem Original RGB
    2. Suavização Gaussiana (NumPy)
    3. Mapa de Matiz HSV
    4. Textura LBP
    5. Magnitude do Gradiente de Sobel
    6. Espectro de Potência 2D de Fourier
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

    # Suavização Gaussiana (do zero em NumPy)
    blurred_bgr = apply_gaussian(img_resized, ksize=5, sigma=1.0)
    blurred_bgr = np.clip(blurred_bgr, 0, 255).astype(np.uint8)
    blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

    # Espaço HSV (visualização do canal Hue como colormap)
    hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    hue_colormap = cv2.applyColorMap((hue_channel * (255 / 180)).astype(np.uint8), cv2.COLORMAP_HSV)
    hue_colormap_rgb = cv2.cvtColor(hue_colormap, cv2.COLOR_BGR2RGB)

    # Escala de Cinza
    gray = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_norm = ((lbp / lbp.max()) * 255).astype(np.uint8) if lbp.max() > 0 else lbp.astype(np.uint8)
    lbp_colormap = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_VIRIDIS)
    lbp_rgb = cv2.cvtColor(lbp_colormap, cv2.COLOR_BGR2RGB)

    # Sobel
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = np.clip((mag / (mag.max() + 1e-6)) * 255, 0, 255).astype(np.uint8)
    sobel_colormap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
    sobel_rgb = cv2.cvtColor(sobel_colormap, cv2.COLOR_BGR2RGB)

    # Fourier 2D
    f_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    magnitude_spectrum = np.log(1 + np.abs(f_shift))
    fft_norm = ((magnitude_spectrum / (magnitude_spectrum.max() + 1e-6)) * 255).astype(np.uint8)
    fft_colormap = cv2.applyColorMap(fft_norm, cv2.COLORMAP_INFERNO)
    fft_rgb = cv2.cvtColor(fft_colormap, cv2.COLOR_BGR2RGB)

    return {
        "original_rgb": img_rgb,
        "blurred_rgb": blurred_rgb,
        "hue_rgb": hue_colormap_rgb,
        "lbp_rgb": lbp_rgb,
        "sobel_rgb": sobel_rgb,
        "fft_rgb": fft_rgb,
        "blurred_bgr": blurred_bgr,
    }


def predict_single_image(img_bgr_blurred: np.ndarray, classical_models: dict, cnn_model: torch.nn.Module):
    """Executa a inferência individual no pipeline clássico e na CNN."""
    # --- Clássico ---
    t0 = time.time()
    feat = extract_features(img_bgr_blurred)
    feat_scaled = classical_models["scaler"].transform(feat.reshape(1, -1))
    svm_pred_idx = int(classical_models["svm"].predict(feat_scaled)[0])
    knn_pred_idx = int(classical_models["knn"].predict(feat_scaled)[0])
    t_classical_ms = (time.time() - t0) * 1000

    knn_probs = classical_models["knn"].predict_proba(feat_scaled)[0]

    # --- CNN ---
    t1 = time.time()
    img_rgb = cv2.cvtColor(img_bgr_blurred, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(tensor)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
    cnn_pred_idx = int(probs.argmax())
    t_cnn_ms = (time.time() - t1) * 1000

    return {
        "svm_pred": CLASS_NAMES[svm_pred_idx],
        "svm_pred_idx": svm_pred_idx,
        "knn_pred": CLASS_NAMES[knn_pred_idx],
        "knn_pred_idx": knn_pred_idx,
        "knn_probs": knn_probs,
        "t_classical_ms": t_classical_ms,
        "cnn_pred": CLASS_NAMES[cnn_pred_idx],
        "cnn_pred_idx": cnn_pred_idx,
        "cnn_probs": probs,
        "t_cnn_ms": t_cnn_ms,
        "features": feat,
    }


# ---------------------------------------------------------------------------
# Interface Principal
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🎮 Classificação de Jogos por Capturas de Tela</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Projeto de Processamento de Imagens</b> — Comparação entre <b>Pipeline Clássico</b> (NumPy, OpenCV, LBP, GLCM, Sobel, Fourier, k-NN/SVM) e <b>Rede Neural Convolucional</b> (MobileNetV2 Transfer Learning).</div>',
    unsafe_allow_html=True,
)

# Carrega modelos e dados
try:
    classical_models, cnn_model = get_models()
    train_df, val_df, test_df = get_splits_data()
    metrics_tables = get_metrics_tables()
except Exception as e:
    st.error(f"Erro ao carregar modelos ou dados do projeto: {e}")
    st.stop()

# Navegação por Abas
tab1, tab2, tab3 = st.tabs([
    "🔍 Comparação Individual por Categoria",
    "🛠️ Passo a Passo do Pipeline",
    "📊 Resultados Finais Globais",
])


# ===========================================================================
# ABA 1: COMPARAÇÃO INDIVIDUAL POR CATEGORIA
# ===========================================================================
with tab1:
    st.markdown("### Comparação Lado a Lado: Clássico vs. CNN em Imagem Individual")
    st.markdown(
        "Selecione uma imagem de teste ou envie uma captura própria para ver as **transformações visuais intermediárias** e a **predição individual de cada método** no mesmo instante."
    )

    input_mode = st.radio(
        "Escolha a origem da imagem para análise:",
        options=["📂 Amostra do Conjunto de Teste (751 imagens não vistas no treino)", "📤 Upload de arquivo próprio (PNG/JPG)"],
        horizontal=True,
    )

    img_bgr = None
    true_label = None
    sample_path_info = ""

    if input_mode.startswith("📂"):
        col_ctrl1, col_ctrl2 = st.columns([2, 3])
        with col_ctrl1:
            selected_category = st.selectbox(
                "🎯 Selecione a Categoria do Teste:",
                options=CLASS_NAMES,
                index=0,
            )

        if test_df is not None:
            class_test_df = test_df[test_df["label"] == selected_category].reset_index(drop=True)
            if len(class_test_df) > 0:
                with col_ctrl2:
                    sample_idx = st.slider(
                        f"Escolha entre as {len(class_test_df)} imagens de teste de '{selected_category}':",
                        min_value=0,
                        max_value=len(class_test_df) - 1,
                        value=0,
                        step=1,
                        key=f"sample_slider_{selected_category}",
                    )
                    selected_row = class_test_df.iloc[sample_idx]
                    true_label = selected_row["label"]
                    sample_path_info = selected_row["path"]

                    # Garante resolução correta de caminho
                    if not os.path.exists(sample_path_info):
                        candidate = os.path.join(PROJECT_ROOT, selected_row["label"], os.path.basename(sample_path_info))
                        if os.path.exists(candidate):
                            sample_path_info = candidate

                    img_bgr = cv2.imread(sample_path_info)
        else:
            st.warning("Splits não encontrados em results/splits/splits.json.")

    else:
        # Modo Upload
        col_up1, col_up2 = st.columns([2, 3])
        with col_up1:
            user_declared_game = st.selectbox(
                "🎯 Qual é o jogo verdadeiro desta imagem enviada?",
                options=["🤖 Deixar a IA adivinhar (Classificação Pura)"] + CLASS_NAMES,
                index=0,
                help="Se você escolher um jogo específico, o sistema verificará se a IA acertou ou errou. Se escolher 'Deixar a IA adivinhar', ela fará a predição direta.",
            )
            if user_declared_game != "🤖 Deixar a IA adivinhar (Classificação Pura)":
                true_label = user_declared_game
            else:
                true_label = None

        with col_up2:
            uploaded_file = st.file_uploader("Envie uma captura de tela (640×360 ou similar):", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                sample_path_info = f"Upload do Usuário: {uploaded_file.name}"

    if img_bgr is not None:
        # Computa transformações e predições
        with st.spinner("Processando transformações e executando inferência..."):
            transforms = compute_visual_transformations(img_bgr)
            preds = predict_single_image(transforms["blurred_bgr"], classical_models, cnn_model)

        # -------------------------------------------------------------
        # 1. Painel Visual das Etapas do Processamento de Imagens
        # -------------------------------------------------------------
        st.markdown("#### 🔬 Etapas Visuais Intermediárias do Pré-Processamento")
        tcol1, tcol2, tcol3, tcol4, tcol5, tcol6 = st.columns(6)

        with tcol1:
            st.image(transforms["original_rgb"], caption="1. Original (224×224)", use_container_width=True)
        with tcol2:
            st.image(transforms["blurred_rgb"], caption="2. Gaussiano NumPy (k=5)", use_container_width=True)
        with tcol3:
            st.image(transforms["hue_rgb"], caption="3. Matiz HSV (Cor)", use_container_width=True)
        with tcol4:
            st.image(transforms["lbp_rgb"], caption="4. Textura LBP (P=8, R=1)", use_container_width=True)
        with tcol5:
            st.image(transforms["sobel_rgb"], caption="5. Bordas Sobel (Gx + Gy)", use_container_width=True)
        with tcol6:
            st.image(transforms["fft_rgb"], caption="6. Fourier 2D (Frequência)", use_container_width=True)

        st.divider()

        # -------------------------------------------------------------
        # 2. Comparação Direta das Predições
        # -------------------------------------------------------------
        if true_label is not None:
            st.markdown(f"#### ⚖️ Comparação das Predições (Rótulo Verdadeiro: **{true_label}**)")
        else:
            st.markdown("#### ⚖️ Comparação das Predições (Classificação Pura da IA)")

        pcol1, pcol2 = st.columns(2)

        # Coluna Clássico
        with pcol1:
            st.markdown("### 🏛️ Pipeline Clássico (109 Features)")

            if true_label is not None:
                svm_correct = (preds["svm_pred"] == true_label)
                knn_correct = (preds["knn_pred"] == true_label)
                badge_svm = f'<span class="badge-correct">✓ ACERTOU ({true_label})</span>' if svm_correct else f'<span class="badge-wrong">✗ CONFUNDIU (Esperado: {true_label})</span>'
                badge_knn = f'<span class="badge-correct">✓ ACERTOU ({true_label})</span>' if knn_correct else f'<span class="badge-wrong">✗ CONFUNDIU (Esperado: {true_label})</span>'
            else:
                svm_correct = None
                knn_correct = None
                badge_svm = '<span class="badge-pred">🤖 PREDIÇÃO</span>'
                badge_knn = '<span class="badge-pred">🤖 PREDIÇÃO</span>'

            st.markdown(
                f"""
                <div class="card-claro">
                    <div style="margin-bottom:12px;">
                        <b>Classificador SVM (Kernel RBF):</b><br>
                        <span style="font-size:1.3rem; font-weight:700; color:#1E3A8A;">{preds["svm_pred"]}</span> &nbsp; {badge_svm}
                    </div>
                    <div style="margin-bottom:12px;">
                        <b>Classificador k-NN (k=5):</b><br>
                        <span style="font-size:1.1rem; font-weight:600; color:#334155;">{preds["knn_pred"]}</span> &nbsp; {badge_knn}
                    </div>
                    <hr style="margin:10px 0;">
                    <div style="font-size:0.9rem; color:#475569;">
                        ⏱️ <b>Tempo de Inferência:</b> <code>{preds["t_classical_ms"]:.2f} ms</code><br>
                        📐 <b>Vetor de Entrada:</b> 109 características (64 HSV + 10 LBP + 16 GLCM + 8 Fourier + 11 Sobel)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probabilidades k-NN
            with st.expander("Ver probabilidades estimadas pelo k-NN (vizinhos mais próximos):"):
                knn_df = pd.DataFrame({
                    "Jogo": CLASS_NAMES,
                    "Probabilidade (%)": preds["knn_probs"] * 100
                }).sort_values(by="Probabilidade (%)", ascending=False)
                st.dataframe(knn_df, use_container_width=True, hide_index=True)

        # Coluna CNN
        with pcol2:
            st.markdown("### 🧠 Rede Neural Convolucional (CNN)")

            confidence = preds["cnn_probs"][preds["cnn_pred_idx"]] * 100

            if true_label is not None:
                cnn_correct = (preds["cnn_pred"] == true_label)
                badge_cnn = f'<span class="badge-correct">✓ ACERTOU ({true_label})</span>' if cnn_correct else f'<span class="badge-wrong">✗ CONFUNDIU (Esperado: {true_label})</span>'
            else:
                cnn_correct = None
                badge_cnn = '<span class="badge-pred">🤖 PREDIÇÃO</span>'

            st.markdown(
                f"""
                <div class="card-claro">
                    <div style="margin-bottom:12px;">
                        <b>MobileNetV2 (Transfer Learning):</b><br>
                        <span style="font-size:1.3rem; font-weight:700; color:#1E3A8A;">{preds["cnn_pred"]}</span> &nbsp; {badge_cnn}
                    </div>
                    <div style="margin-bottom:12px;">
                        <b>Grau de Confiança (Softmax):</b><br>
                        <span style="font-size:1.1rem; font-weight:600; color:#047857;">{confidence:.2f}%</span>
                    </div>
                    <hr style="margin:10px 0;">
                    <div style="font-size:0.9rem; color:#475569;">
                        ⏱️ <b>Tempo de Inferência (CPU):</b> <code>{preds["t_cnn_ms"]:.2f} ms</code><br>
                        🧬 <b>Parâmetros:</b> Backbone ImageNet (congelado) + Camada Densa 1280→10
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Gráfico de barras com as probabilidades da CNN
            with st.expander("Ver distribuição de probabilidades de todas as classes (CNN):", expanded=True):
                prob_df = pd.DataFrame({
                    "Jogo": CLASS_NAMES,
                    "Confiança (%)": preds["cnn_probs"] * 100
                }).sort_values(by="Confiança (%)", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3.2))
                highlight_target = true_label if true_label is not None else preds["cnn_pred"]
                colors = ["#22C55E" if j == highlight_target else "#94A3B8" for j in prob_df["Jogo"]]
                ax.barh(prob_df["Jogo"], prob_df["Confiança (%)"], color=colors)
                ax.set_xlabel("Confiança (%)", fontsize=8)
                ax.set_xlim(0, 100)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # -------------------------------------------------------------
        # 3. Card de Diagnóstico e Comparação Relativa
        # -------------------------------------------------------------
        ratio_speed = preds["t_cnn_ms"] / max(preds["t_classical_ms"], 1e-3)
        agreement = (preds["svm_pred"] == preds["cnn_pred"])

        if true_label is not None:
            if (preds["svm_pred"] == true_label) and (preds["cnn_pred"] == true_label):
                diag_msg = f"🏆 <b>Sucesso Total:</b> Ambos os modelos acertaram com precisão (identificaram corretamente como <b>{true_label}</b>)."
            elif (preds["svm_pred"] == true_label) and (preds["cnn_pred"] != true_label):
                diag_msg = f"⚡ <b>Vitória do Clássico:</b> O SVM acertou (<b>{true_label}</b>), enquanto a CNN confundiu com <b>{preds['cnn_pred']}</b>."
            elif (preds["svm_pred"] != true_label) and (preds["cnn_pred"] == true_label):
                diag_msg = f"🧠 <b>Vitória da CNN:</b> A CNN acertou (<b>{true_label}</b>), enquanto o SVM confundiu com <b>{preds['svm_pred']}</b>."
            else:
                diag_msg = f"⚠️ <b>Ambos Divergiram do Esperado:</b> Rótulo esperado: <b>{true_label}</b> | SVM prediz: <b>{preds['svm_pred']}</b> | CNN prediz: <b>{preds['cnn_pred']}</b>."
        else:
            diag_msg = f"🤖 <b>Modo de Classificação Pura:</b> SVM prediz <b>{preds['svm_pred']}</b> | CNN prediz <b>{preds['cnn_pred']}</b> (com {confidence:.1f}% de certeza)."

        st.markdown(
            f"""
            <div class="card-diagnostico">
                💡 <b>Diagnóstico da Comparação:</b><br>
                • <b>Avaliação:</b> {diag_msg}<br>
                • <b>Concordância:</b> {'Ambos os modelos concordaram na mesma predição.' if agreement else 'Os modelos divergiram entre si.'}<br>
                • <b>Velocidade Relativa:</b> O pipeline clássico (SVM) executou aproximadamente <b>{ratio_speed:.1f}x mais rápido</b> que o forward pass da CNN nesta imagem.<br>
                • <b>Origem da Amostra:</b> <code>{sample_path_info}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # 4. Inspecionar o Vetor de Características (109 Dimensões)
        # -------------------------------------------------------------
        with st.expander("📊 Inspecionar o Vetor de 109 Características Clássicas Desta Imagem"):
            names = feature_names()
            feat_df = pd.DataFrame({"Característica": names, "Valor": preds["features"]})

            fcol1, fcol2 = st.columns([1, 1])
            with fcol1:
                st.markdown("**Valores Numéricos (Primeiros 30 atributos):**")
                st.dataframe(feat_df.head(30), use_container_width=True, height=250)
            with fcol2:
                st.markdown("**Resumo por Bloco de Processamento:**")
                st.write({
                    "HSV (Matiz e Saturação)": f"{feat_df.iloc[0:64]['Valor'].sum():.3f} (Soma das frequências)",
                    "LBP (Microtextura Uniforme)": f"{feat_df.iloc[64:74]['Valor'].sum():.3f} (Soma normalizada)",
                    "GLCM (Contraste/Energia de Cinza)": f"Média = {feat_df.iloc[74:90]['Valor'].mean():.4f}",
                    "Fourier 2D (Energia Espectral Radial)": f"{feat_df.iloc[90:98]['Valor'].sum():.3f} (Fração de energia)",
                    "Gradientes Sobel (Bordas e Orientação)": f"Densidade = {feat_df.iloc[100]['Valor']:.4f}",
                })
    else:
        st.info("Envie uma imagem ou certifique-se de selecionar uma categoria válida.")


# ===========================================================================
# ABA 2: PASSO A PASSO DO PIPELINE
# ===========================================================================
with tab2:
    st.markdown("### 🛠️ Passo a Passo Metodológico do Projeto")
    st.markdown(
        "O projeto foi estruturado em duas abordagens rigorosamente comparadas sobre o mesmo conjunto de teste de 751 imagens:"
    )

    col_step1, col_step2 = st.columns([1, 1])

    with col_step1:
        st.markdown(
            """
            <div class="step-box">
                <h4>1. Aquisição e Redimensionamento</h4>
                • Leitura em disco via OpenCV (<code>cv2.imread</code>).<br>
                • Redimensionamento uniforme para <b>224×224 pixels</b> (<code>cv2.INTER_AREA</code>), mantendo proporção dimensional estável entre ambos os métodos.
            </div>
            <div class="step-box">
                <h4>2. Suavização Gaussiana (Implementação do Zero em NumPy)</h4>
                • Implementado do zero sem <code>cv2.GaussianBlur</code>.<br>
                • Utiliza a <b>propriedade de separabilidade</b> do kernel 2D em duas convoluções 1D ortogonais (linhas e colunas).<br>
                • Reduz a complexidade de <code>O(N · k²)</code> para <code>O(N · k)</code> e elimina ruídos de compressão.
            </div>
            <div class="step-box">
                <h4>3. Processamento no Espaço de Cores (HSV)</h4>
                • Conversão BGR → HSV para desacoplar cromaticidade de iluminação.<br>
                • <b>32 bins de Matiz (H)</b> + <b>32 bins de Saturação (S)</b> = 64 dimensões normalizadas, capturando paletas exclusivas de cada jogo.
            </div>
            <div class="step-box">
                <h4>4. Descritores de Textura e Bordas</h4>
                • <b>LBP uniforme (10 dims)</b>: microestrutura e invariância à iluminação monotônica ($P=8, R=1$).<br>
                • <b>GLCM Haralick (16 dims)</b>: contraste, homogeneidade, energia e correlação em 4 direções.<br>
                • <b>Bordas Sobel (11 dims)</b>: magnitude média, desvio padrão, densidade de arestas e histograma de ângulos em 8 bins.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_step2:
        st.markdown(
            """
            <div class="step-box">
                <h4>5. Transformada de Fourier 2D (FFT)</h4>
                • Transformada Rápida de Fourier 2D centralizada (<code>np.fft.fft2 + fftshift</code>).<br>
                • Partição do espectro em <b>8 anéis radiais concêntricos</b>.<br>
                • Mede a fração de energia de baixas frequências (superfícies lisas) vs. altas frequências (detalhes finos e pixel art).
            </div>
            <div class="step-box">
                <h4>6. Classificação Clássica (k-NN e SVM)</h4>
                • Vetor consolidado de <b>109 características</b> normalizado com <code>StandardScaler</code>.<br>
                • <b>k-NN ($k=5$)</b>: classificador baseado em instâncias.<br>
                • <b>SVM (RBF, $C=10$)</b>: separação em espaço de alta dimensão com margem máxima.
            </div>
            <div class="step-box">
                <h4>7. Abordagem B: CNN (MobileNetV2 Transfer Learning)</h4>
                • Backbone pré-treinado no ImageNet com parâmetros <b>100% congelados</b>.<br>
                • Classificador customizado (Dropout 0.2 + Linear 1280→10, apenas 12.810 parâmetros treináveis).<br>
                • Data Augmentation (flip horizontal e rotação ±10°) com otimizador Adam e Early Stopping.
            </div>
            <div class="step-box">
                <h4>8. Avaliação Sob o Mesmo Conjunto de Teste</h4>
                • Divisão estratificada 70% Treino, 15% Validação, 15% Teste (751 imagens fixas).<br>
                • Garantia total de ausência de <i>data leakage</i>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🖼️ Visualizações Globais das Características do Dataset")

    tab_vis1, tab_vis2, tab_vis3, tab_vis4 = st.tabs([
        "Grid das 10 Classes Pré-processadas",
        "Assinatura de Cor HSV",
        "Densidade de Bordas vs Textura",
        "Espectro Radial de Fourier",
    ])

    with tab_vis1:
        grid_path = os.path.join(PROJECT_ROOT, "results", "figures", "sample_games_grid.png")
        if os.path.exists(grid_path):
            st.image(grid_path, caption="Amostras dos 10 Jogos após Redimensionamento e Suavização Gaussiana (NumPy)", use_container_width=True)

    with tab_vis2:
        hsv_path = os.path.join(PROJECT_ROOT, "results", "figures", "feature_hsv_distributions.png")
        if os.path.exists(hsv_path):
            st.image(hsv_path, caption="Perfil Médio de Matiz (Hue) no Espaço HSV para cada um dos 10 Jogos", use_container_width=True)

    with tab_vis3:
        edge_path = os.path.join(PROJECT_ROOT, "results", "figures", "feature_edge_texture_comparison.png")
        if os.path.exists(edge_path):
            st.image(edge_path, caption="Densidade Média de Bordas (Sobel) e Contraste de Textura (GLCM) por Jogo", use_container_width=True)

    with tab_vis4:
        four_path = os.path.join(PROJECT_ROOT, "results", "figures", "feature_fourier_spectrum.png")
        if os.path.exists(four_path):
            st.image(four_path, caption="Assinatura de Frequência Espacial (Fourier 2D) por Jogo ao Longo de 8 Faixas Radiais", use_container_width=True)


# ===========================================================================
# ABA 3: RESULTADOS FINAIS GLOBAIS
# ===========================================================================
with tab3:
    st.markdown("### 📊 Resultados Finais e Comparação Quantitativa")
    st.markdown(
        "Desempenho consolidado de todas as abordagens avaliadas sobre as **mesmas 751 capturas de tela** do conjunto de teste independente:"
    )

    # Cards de Métricas Principais
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">k-NN (Clássico)</div>
                <div class="metric-value">88,81%</div>
                <div style="font-size:0.85rem; color:#64748B;">F1-macro: 0,8887 | 1,58 ms/img</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">SVM RBF (Clássico)</div>
                <div class="metric-value">94,54%</div>
                <div style="font-size:0.85rem; color:#16A34A; font-weight:600;">F1-macro: 0,9452 | 0,35 ms/img</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">CNN MobileNetV2</div>
                <div class="metric-value">95,74%</div>
                <div style="font-size:0.85rem; color:#64748B;">F1-macro: 0,9571 | 15,88 ms/img</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Vantagem de Velocidade</div>
                <div class="metric-value">45×</div>
                <div style="font-size:0.85rem; color:#64748B;">SVM mais rápido que CNN em CPU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabela Resumo Lado a Lado
    summary_df = pd.DataFrame({
        "Método": ["k-NN (k=5)", "SVM (Kernel RBF)", "CNN (MobileNetV2)"],
        "Paradigma": ["Clássico (109 features)", "Clássico (109 features)", "Deep Learning (Transfer Learning)"],
        "Acurácia Top-1": ["88,81%", "94,54%", "95,74%"],
        "F1-macro": [0.8887, 0.9452, 0.9571],
        "Tempo Total (751 imagens)": ["1.186,7 ms", "264,5 ms", "11.929,6 ms"],
        "Tempo Médio por Imagem": ["~1,58 ms", "~0,35 ms", "~15,88 ms"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # Gráfico Comparativo e Curvas de Treino
    rcol1, rcol2 = st.columns([1, 1])

    with rcol1:
        st.markdown("#### 📈 Gráfico Comparativo: Clássico vs CNN")
        comp_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "comparison_methods.png")
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="Acurácia e F1-macro para cada método no conjunto de teste", use_container_width=True)

    with rcol2:
        st.markdown("#### 📉 Evolução do Treinamento da CNN (MobileNetV2)")
        hist_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "cnn_training_history.png")
        if os.path.exists(hist_img_path):
            st.image(hist_img_path, caption="Curvas de Perda (Loss) e Acurácia ao longo das épocas", use_container_width=True)

    st.divider()

    # Matrizes de Confusão
    st.markdown("#### 🎯 Matrizes de Confusão (Normalizadas por Recall)")
    cm_tab1, cm_tab2, cm_tab3 = st.tabs(["SVM (Clássico)", "CNN (MobileNetV2)", "k-NN (Clássico)"])

    with cm_tab1:
        cm_svm = os.path.join(PROJECT_ROOT, "results", "figures", "confusion_matriz_de_confusao_-_svm.png")
        if os.path.exists(cm_svm):
            st.image(cm_svm, caption="Matriz de Confusão do SVM (Acurácia: 94,54%)", use_container_width=True)

    with cm_tab2:
        cm_cnn = os.path.join(PROJECT_ROOT, "results", "figures", "confusion_matriz_de_confusao_-_cnn.png")
        if os.path.exists(cm_cnn):
            st.image(cm_cnn, caption="Matriz de Confusão da CNN (Acurácia: 95,74%)", use_container_width=True)

    with cm_tab3:
        cm_knn = os.path.join(PROJECT_ROOT, "results", "figures", "confusion_matriz_de_confusao_-_k-nn.png")
        if os.path.exists(cm_knn):
            st.image(cm_knn, caption="Matriz de Confusão do k-NN (Acurácia: 88,81%)", use_container_width=True)

    st.divider()

    # Relatório Detalhado por Classe
    st.markdown("#### 📋 Relatório Detalhado de Métricas por Classe")
    if "svm" in metrics_tables and "cnn" in metrics_tables:
        df_svm = metrics_tables["svm"].loc[CLASS_NAMES, ["precision", "recall", "f1-score"]].copy()
        df_cnn = metrics_tables["cnn"].loc[CLASS_NAMES, ["precision", "recall", "f1-score"]].copy()

        df_combined = pd.DataFrame({
            "SVM Precisão": df_svm["precision"],
            "SVM Recall": df_svm["recall"],
            "SVM F1": df_svm["f1-score"],
            "CNN Precisão": df_cnn["precision"],
            "CNN Recall": df_cnn["recall"],
            "CNN F1": df_cnn["f1-score"],
        })
        st.dataframe(df_combined.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["SVM F1", "CNN F1"]), use_container_width=True)

    # Análise Conclusiva
    st.markdown(
        """
        <div class="card-conclusao">
            <b>📌 Conclusões Principais do Estudo:</b><br>
            1. <b>Competitividade do Clássico:</b> O SVM alimentado com o vetor de 109 características atingiu <b>94,54% de acurácia</b>, ficando a apenas <b>1,2 ponto percentual</b> da CNN pré-treinada.<br>
            2. <b>Eficiência Extrema:</b> O tempo de inferência do SVM em CPU foi de apenas <b>~0,35 ms por imagem</b> (45× mais rápido que os 15,88 ms da CNN), ideal para aplicações em tempo real e equipamentos de borda.<br>
            3. <b>Interpretabilidade:</b> Padrões de acerto e confusão são completamente explicáveis pelos descritores visuais (paleta HSV e geometria de bordas e texturas).
        </div>
        """,
        unsafe_allow_html=True,
    )