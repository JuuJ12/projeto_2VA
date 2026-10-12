# Usar 224×224 para ambos os pipelines, abandonando a proporção 16:9 da proposta

A proposta inicial especificava redimensionamento mantendo a proporção 16:9 (ex.: 224×126). Decidimos usar 224×224 (quadrado) para o pipeline clássico e a CNN, porque MobileNetV2 exige entrada quadrada e manter resoluções diferentes introduziria uma variável de confusão na comparação. A distorção da proporção é irrelevante para discriminar estilo visual entre jogos.

## Considered Options

- **224×224 para ambos** ← escolhido
- 224×126 para clássico + 224×224 para CNN: preserva 16:9 no clássico mas invalida a comparação direta de tempo de inferência e impõe dois caminhos de pré-processamento
- 256×144 com crop central 224×224: mais correto geometricamente, mas adiciona complexidade sem benefício mensurável para classificação
