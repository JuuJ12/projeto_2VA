# Usar transfer learning (backbone congelado) em vez de CNN treinada do zero

Optamos por MobileNetV2 pré-treinado no ImageNet com backbone congelado, treinando apenas o classificador linear (1280→10). Treinar do zero exigiria GPU e centenas de épocas para ser competitivo, inviável no prazo de 5 semanas. O objetivo do projeto é demonstrar a vantagem do deep learning sobre o pipeline clássico, não estudar o processo de treinamento da CNN em si; transfer learning cumpre esse objetivo de forma confiável.

## Considered Options

- **Transfer learning, backbone congelado** ← escolhido
- CNN pequena treinada do zero (3–5 camadas conv): mais "honesta" pedagogicamente, mas resultado provavelmente inferior ao pipeline clássico em 5.000 imagens, o que inverteria a narrativa do projeto
- Fine-tuning completo (backbone descongelado, lr baixo): melhor resultado final, mas requer GPU e gestão de overfitting — complexidade desnecessária para o prazo

## Consequences

O termo **fine-tuning** está reservado, no glossário, para quando o backbone também é ajustado. O que fazemos aqui é estritamente **transfer learning**.
