# Identificação Automática de Jogos Eletrônicos por Capturas de Tela

> **Comparação entre Processamento Clássico de Imagens e Redes Neurais Convolucionais (CNN)**  
> Disciplina de Processamento de Imagens — Projeto 2VA  

---

## 🎮 Visão Geral

Este projeto aborda o problema de **classificação automática de capturas de tela (screenshots) de gameplay** em um de **10 títulos populares de jogos eletrônicos**. O problema tem grande relevância prática para plataformas de streaming (Twitch, YouTube Gaming), lojas digitais (Steam, Epic Games Store) e bases de dados que necessitam organizar, indexar e moderar grandes volumes de mídia de forma automatizada.

O projeto compara diretamente duas abordagens concorrentes avaliadas sob o **mesmo conjunto de teste independente (750 imagens)**:
1. **Pipeline Clássico**: Suavização Gaussiana implementada do zero em NumPy, extração manual de vetor de 109 características (Cores HSV, Textura LBP, Coocorrência GLCM, Estatísticas de Bordas Sobel e Espectro Radial de Fourier 2D) com classificação via k-NN e SVM (RBF).
2. **Deep Learning (CNN)**: Rede convolucional MobileNetV2 utilizando Transfer Learning com backbone pré-treinado no ImageNet (congelado) e camada de classificação ajustada com data augmentation e early stopping.

---

## 🕹️ Classes do Dataset (10 Jogos)

O dataset utilizado é o **Bingsu/Gameplay_Images**, contendo 10.000 capturas de tela em formato PNG (640×360):
1. **Among Us** (Arte minimalista 2D vetorial, áreas planas de cor sólida)
2. **Apex Legends** (FPS 3D fotorrealista, alta densidade de detalhes de combate)
3. **Fortnite** (Estilo estilizado/cartoon 3D, paleta viva e vibrante)
4. **Forza Horizon** (Simulação automobilística hiper-realista, iluminação natural e reflexos)
5. **Free Fire** (Battle Royale mobile 3D com texturas otimizadas)
6. **Genshin Impact** (Estilo anime/cel-shaded 3D, cenários abertos coloridos)
7. **God of War** (Gráficos cinematográficos AAA, paleta sóbria, texturas orgânicas)
8. **Minecraft** (Mundo voxel 3D, blocos cúbicos com bordas ortogonais marcadas)
9. **Roblox** (Gráficos 3D poliédricos variados e lúdicos)
10. **Terraria** (Pixel art 2D side-scrolling, alta frequência de bordas e grade de pixels)

O subconjunto de trabalho utiliza **500 imagens por classe (5.000 imagens no total)** com partição estratificada:
- **Treino**: 70% (~3.500 imagens)
- **Validação**: 15% (750 imagens)
- **Teste**: 15% (750 imagens)

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Linguagem**: Python 3.10+
- **Processamento de Imagens e Álgebra Linear**:
  - `numpy`: Convoluções espaciais 1D separáveis do zero, FFT 2D radial e manipulações matriciais.
  - `opencv-python` (`cv2`): Leitura de imagens, redimensionamento, conversões de espaço de cor (BGR/HSV/Cinza), gradiente Sobel e histogramas.
  - `scikit-image`: Extração de Local Binary Patterns (LBP) e matrizes de coocorrência de níveis de cinza (GLCM).
- **Classificação e Aprendizado de Máquina**:
  - `scikit-learn`: Normalização (`StandardScaler`), classificadores `KNeighborsClassifier` e `SVC`, matrizes de confusão e métricas de avaliação (`f1_score`, `classification_report`).
- **Deep Learning**:
  - `torch` e `torchvision`: MobileNetV2 com transfer learning, data augmentation e otimizador Adam.
- **Visualização**:
  - `matplotlib` e `pandas`: Gráficos comparativos, curvas de treino e matrizes de confusão.

---

## 📁 Estrutura do Projeto

```text
projeto_2VA/
├── app.py                      # Interface Web interativa em Streamlit
├── main.py                     # CLI principal para executar os experimentos
├── GLOSSARY.md                 # Glossário canônico de termos do projeto
├── README.md                   # Este arquivo (tutorial e documentação)
├── docs/
│   └── adr/                    # Architecture Decision Records (ADRs)
│       ├── 0001-resolucao-224x224-uniforme.md
│       ├── 0002-transfer-learning-backbone-congelado.md
│       ├── 0003-subconjunto-500-por-classe.md
│       └── 0004-escopo-implementacao-do-zero.md
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Amostragem estratificada e persistência dos splits
│   ├── preprocessing.py        # Redimensionamento e Convolução Gaussiana do zero (NumPy)
│   ├── features.py             # Extração das 109 características (HSV, LBP, GLCM, Bordas, Fourier)
│   ├── classical.py            # Treinamento e inferência de k-NN e SVM
│   ├── cnn.py                  # MobileNetV2, DataLoaders, loop de treino e inferência
│   ├── evaluate.py             # Métricas (acurácia, F1-macro), matrizes de confusão e erros
│   └── visualize.py            # Gráficos de análise das características por jogo
├── results/
│   ├── figures/                # Matrizes de confusão e gráficos comparativos gerados
│   ├── metrics/                # Relatórios de classificação por classe em CSV
│   ├── models/                 # Modelos salvos (.pkl, .pt)
│   └── splits/                 # Partição treino/val/teste salva em JSON
└── report/
    ├── RELATORIO.md            # Relatório técnico completo do projeto
    └── ROTEIRO_APRESENTACAO.md # Estrutura de slides e roteiro do vídeo de 10 min
```

---

## 🚀 Tutorial de Execução

### 1. Clonar o repositório e preparar o ambiente
```bash
git clone <url-do-repositorio>
cd projeto_2VA
```

### 2. Instalar as dependências
```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib pandas torch torchvision streamlit
```

### 3. Executar a Interface Gráfica Interativa (Streamlit)

Para abrir a interface visual completa com comparação individual por imagem e resultados:
```bash
streamlit run app.py
```

### 4. Executar os Experimentos via Linha de Comando (CLI)

#### Apenas Pipeline Clássico (k-NN e SVM):
```bash
python main.py --method classical
```

#### Apenas Rede Neural Convolucional (CNN - MobileNetV2):
```bash
python main.py --method cnn --epochs 6
```

#### Ambos os Métodos com Comparação Lado a Lado:
```bash
python main.py --method both --epochs 6
```

#### Forçar Re-treinamento Completo:
```bash
python main.py --method both --retrain --epochs 6
```

#### Gerar Gráficos de Características por Jogo:
```bash
python src/visualize.py
```

---

## 📊 Destaques Metodológicos

### Implementação do Zero (NumPy)
A suavização espacial utiliza **convolução Gaussiana 1D separável implementada manualmente em NumPy** (`src/preprocessing.py`). Ao invés de uma convolução 2D ingênua $O(N \cdot k^2)$, o filtro decompõe o kernel 2D em dois passos 1D ortogonais ($O(N \cdot k)$), proporcionando alta eficiência e precisão idêntica ao OpenCV.

### Vetor de 109 Características Clássicas
- **Cor (64 dims)**: Histograma HSV normalizado (32 bins Matiz + 32 bins Saturação).
- **Textura Microestrutural (10 dims)**: Histograma de Local Binary Patterns (LBP) uniforme com $P=8, R=1$.
- **Textura de Segunda Ordem (16 dims)**: Matriz de Coocorrência (GLCM) calculando contraste, homogeneidade, energia e correlação em 4 direções ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
- **Bordas e Gradientes (11 dims)**: Magnitude média Sobel, desvio padrão da magnitude, densidade de pixels de borda e histograma de orientações de gradiente em 8 faixas.
- **Frequência Espacial (8 dims)**: Energia espectral radial calculada a partir da Transformada Rápida de Fourier 2D (`np.fft.fft2`).

---

## 👥 Autores
- Alunos da disciplina de Processamento de Imagens
