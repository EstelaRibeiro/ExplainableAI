# Score-CAM e Score-CAM Otimizado: Interpretabilidade Livre de Gradientes

**Arquivos no projeto:** [`scorecam.py`](/scorecam.py) e [`scorecam_otimizado.py`](/scorecam_otimizado.py)  
**Artigo pioneiro:** *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks* (Wang, Wang, Du, Yang, Zhang, Ding, Huang & Li, IEEE CVPR Workshops 2020 - arXiv:1910.01279)

---

## 1. Intuição Didática

Todos os métodos anteriores da família CAM (Grad-CAM, Grad-CAM++, Layer-CAM) dependem dos **gradientes** da rede neural. No entanto, em redes profundas, gradientes podem ser enganosos:
- Eles sofrem de **saturação** (quando uma ativação atinge um platô, a derivada vai a zero, mesmo que aquela característica seja crucial).
- Eles representam apenas uma **sensibilidade infinitesimal local**, e não o impacto real da característica na decisão final.

O **Score-CAM** propõe uma mudança radical de paradigma:
> *"Esqueça os gradientes! Vamos tratar cada mapa de ativação da última camada como uma 'máscara visual'. Aplicamos essa máscara sobre a foto original (deixando visível apenas o que aquele canal enxerga) e perguntamos à rede: 'Qual a probabilidade que você prevê agora?'. Quanto maior for a pontuação da rede para essa imagem mascarada, mais importante é aquele canal convolucional."*

Dessa forma, o Score-CAM mede o **aumento direto de confiança (Forward Score)** gerado por cada canal, sem qualquer ruído de retropropagação.

---

## 2. Fundamentação Matemática

Seja $X$ a imagem de entrada e $A_l^k$ o $k$-ésimo canal da camada convolucional $l$.

### Passo 1: Interpolação e Normalização das Máscaras
Cada canal $A_l^k$ é ampliado para o tamanho $(H, W)$ da imagem de entrada e normalizado no intervalo $[0, 1]$:

$$M_k = \frac{\text{Up}(A_l^k) - \min(\text{Up}(A_l^k))}{\max(\text{Up}(A_l^k)) - \min(\text{Up}(A_l^k)) + \epsilon}$$

### Passo 2: Mascaramento da Imagem e Inferência Direta
A imagem original é mascarada pelo produto de Hadamard (pixel por pixel):

$$X_k = X \odot M_k$$

A imagem mascarada $X_k$ é submetida a um passe para a frente (*forward pass*) na rede neural para obter a pontuação do logit $S_c(X_k)$.

### Passo 3: Ponderação via Softmax
Os pesos dos canais $\alpha_k^c$ são obtidos aplicando a função Softmax sobre todos os scores calculados:

$$\alpha_k^c = \frac{\exp(S_c(X_k))}{\sum_{j=1}^{C} \exp(S_c(X_j))}$$

### Passo 4: Combinação Linear e Heatmap
O mapa Score-CAM final é a soma ponderada das ativações originais:

$$L_{\text{Score-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{C} \alpha_k^c A_l^k \right)$$

Após interpolação para $(H, W)$, o mapa é normalizado entre 0 e 1.

---

## 3. O Desafio Computacional e a Otimização no Projeto

### Versão Padrão ([`scorecam.py`](/scorecam.py)):
Na ResNet50, a última camada convolucional possui $C = 2048$ canais. A versão básica do Score-CAM executa um laço `for` passando imagem por imagem:
```python
# Lento: 2048 inferências sequenciais de tamanho 1
for c in range(C):
    score = self.model(masked_image)
    weights.append(score)
```
Isso pode levar dezenas de segundos ou até minutos para uma única imagem em GPU.

### Versão Otimizada ([`scorecam_otimizado.py`](/scorecam_otimizado.py)):
A classe `ScoreCAMotim` implementada no projeto resolve esse gargalo através de **Vetorização e Batching Paralelo na GPU**:

```python
# 1. Normalização vetorizada de todos os canais simultaneamente em tensores 4D
mins = activation_maps.amin(dim=(2, 3), keepdim=True)
maxs = activation_maps.amax(dim=(2, 3), keepdim=True)
masks = (activation_maps - mins) / (maxs - mins + 1e-8)

# 2. Expansão da imagem e multiplicação em bloco
images = image.expand(C, -1, -1, -1)
masked_images = images * masks

# 3. Inferência em lotes (batch_size=64 por padrão)
with torch.no_grad():
    for i in range(0, C, self.batch_size):
        batch = masked_images[i:i+self.batch_size]
        scores = self.model(batch)
        weights.append(scores.squeeze())
```

> **Ganho de Desempenho:** A versão otimizada reduz o tempo de geração do Score-CAM de mais de 40 segundos para cerca de 2 a 3 segundos na GPU, tornando viável a execução em larga escala. É esta versão otimizada que o pipeline [`main_xai_run_all.py`](/main_xai_run_all.py) utiliza!

---

## 4. Vantagens e Limitações

### Vantagens:
- **Zero Ruído de Gradiente:** Imune a problemas de explosão, desaparecimento ou saturação de gradientes.
- **Alta Fidelidade Causal:** Mede diretamente o que acontece na saída da rede quando certas áreas são suprimidas.
- **Passa nos "Sanity Checks":** Demonstra altíssima fidelidade às classes e aos parâmetros da rede.

### Limitações:
- **Demanda Computacional:** Mesmo otimizado com batching, requer múltiplas passagens forward ($C / \text{batch\_size}$ iterações), consumindo mais memória e tempo do que métodos de retropropagação única (como Grad-CAM).
- **Dependência do Tamanho de $C$:** Modelos com camadas convolucionais muito largas (ex.: 2048 filtros) exigem mais VRAM da placa de vídeo.

---

## 5. Como Usar

```python
import torch
from scorecam_otimizado import ScoreCAMotim
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Instanciação com batch_size otimizado para GPU
explainer = ScoreCAMotim(model, device, batch_size=64)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Score-CAM Otimizado", show=True)
```
