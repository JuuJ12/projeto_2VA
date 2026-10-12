# Implementar do zero apenas Gaussiana e Fourier (np.fft); demais via bibliotecas

O professor aprecia implementações do zero (critério subjetivo, não obrigatório). Decidimos implementar manualmente: (1) a **convolução Gaussiana** em NumPy (kernel + operação 2D) e (2) o espectro de Fourier via `np.fft.fft2` (NumPy puro). LBP, GLCM, k-NN e SVM usam scikit-image/scikit-learn.

A razão é custo-benefício: a Gaussiana é curta, didática e justificável (conteúdo de Filtragem); `np.fft` cobre o conteúdo de Fourier sem reimplementar a FFT (que é complexa e não agrega ao entendimento do aluno). LBP e GLCM do zero consumiriam semanas sem ganho pedagógico proporcional.

## Consequences

- Na apresentação, dedicar tempo explicando a implementação da Gaussiana (kernel, padding, convolução 2D) e a interpretação do espectro radial de Fourier — são os pontos "do zero" que o professor vai questionar.
- Não alegar que LBP ou GLCM foram implementados do zero.
