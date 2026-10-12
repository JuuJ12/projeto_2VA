# Trabalhar com subconjunto de 500 imagens por classe em vez do dataset completo

O dataset Bingsu/Gameplay_Images tem 1.000 screenshots por classe (10.000 total). Decidimos trabalhar com 500/classe (5.000 total) como **subconjunto de trabalho** padrão, com o parâmetro `--n-per-class` permitindo escalar para o dataset completo. A razão é pragmática: treino e extração de features rodam em tempo aceitável em CPU (~minutos), tornando os experimentos reproduzíveis em qualquer máquina sem GPU. O subconjunto é selecionado aleatoriamente com semente fixa por classe (split estratificado), garantindo representatividade.

## Consequences

- Todos os resultados e métricas do relatório referem-se ao subconjunto de 5.000, não ao dataset completo de 10.000.
- O parâmetro `--n-per-class` deve ser documentado no README para que avaliadores possam reproduzir com o dataset completo se desejarem.
