# SmoothGrad: Removendo Ruído através da Suavização Estocástica

**Arquivo no projeto:** [`smoothgrad.py`](/smoothgrad.py)  
**Artigo pioneiro:** *SmoothGrad: removing noise by adding noise* (Smilkov, Thorat, Kim, Viégas & Wattenberg, Google Brain, 2017 - arXiv:1706.03825)

---

## 1. Intuição Didática

Imagine tirar uma fotografia de uma paisagem em uma noite muito escura com a sensibilidade ISO da câmera no máximo: a imagem resultante fica cheia de granulados e pontos brancos aleatórios (ruído). Se você tirar 30 fotos idênticas seguidas e calcular a **média de todas elas**, o ruído aleatório em cada foto se cancela mutuamente, e os detalhes reais da cena se tornam nítidos e claros.

O **SmoothGrad** aplica exatamente essa sabedoria contra o maior defeito dos mapas de gradiente: o ruído visual.
> *"Para eliminar o ruído visual espúrio do gradiente, adicione pequenas variações aleatórias de ruído gaussiano à imagem de entrada, gere o mapa de relevância para cada versão perturbada e tire a média final."*

Os pixels que são verdadeiramente importantes para o diagnóstico (como a curvatura aórtica) mantêm gradientes consistentes e construtivos ao longo das perturbações, enquanto o ruído aleatório se anula na média.

---

## 2. Formulação Matemática

Seja $M(x)$ o mapa de sensibilidade produzido por um método base (por exemplo, Vanilla Saliency ou Guided Backprop) para uma imagem $x$. O mapa suavizado **SmoothGrad** $\hat{M}(x)$ é definido pela média amostral:

$$\hat{M}(x) = \frac{1}{N} \sum_{i=1}^{N} M(x + \mathcal{N}(0, \sigma^2))$$

Onde:
- $N$ é o número de amostras perturbadas (no projeto, padrão: $N = 30$).
- $\mathcal{N}(0, \sigma^2)$ é o ruído Gaussiano adicionado a cada pixel.
- $\sigma$ é o desvio padrão do ruído, definido proporcionalmente à escala dinâmica da imagem:
  $$\sigma = \text{noise\_level} \times (x_{\max} - x_{\min})$$
  (no projeto, padrão: $\text{noise\_level} = 0.15$, ou seja, 15% da amplitude).

---

## 3. Fluxo de Execução Passo a Passo

```
                     [ Imagem de Entrada x ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[ Amostra 1: x + ruído_1 ]                     [ Amostra N: x + ruído_N ]
        │                                               │
        ▼                                               ▼
[ Mapa_1 = Explainer(x + ruído_1) ]            [ Mapa_N = Explainer(x + ruído_N) ]
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
              [ Média: (Mapa_1 + ... + Mapa_N) / N ]
                                │
                                ▼
                    [ Normalização [0, 1] ]
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`smoothgrad.py`](/smoothgrad.py):

```python
class SmoothGrad:
    def __init__(self, explainer, device, num_samples=30, noise_level=0.15):
        self.explainer = explainer
        self.device = device
        self.num_samples = num_samples
        self.noise_level = noise_level

    def generate(self, image):
        image = image.to(self.device)
        all_maps = []

        # Executa o explainer na imagem original para preservar métricas de predição
        original_result = self.explainer.generate(image)

        for _ in range(self.num_samples):
            # 1. Gera ruído gaussiano com a mesma dimensão da imagem
            noise = torch.randn_like(image) * self.noise_level
            noisy_image = image + noise

            # 2. Computa a explicação para a imagem ruidosa
            result = self.explainer.generate(noisy_image)
            heatmap = result["heatmap"] if isinstance(result, dict) else result
            all_maps.append(heatmap)

        # 3. Empilha todos os mapas e calcula a média
        smooth_map = torch.stack(all_maps).mean(dim=0)

        # 4. Normaliza para a faixa [0, 1]
        smooth_map = normalize(smooth_map)

        return {
            "heatmap": smooth_map.cpu(),
            "logit": original_result["logit"],
            "probability": original_result["probability"],
            "prediction": original_result["prediction"],
        }
```

### Flexibilidade Modular ("Wrapper Pattern"):
O SmoothGrad neste projeto foi projetado como um *wrapper* (empacotador): ele pode envelopar tanto o [`Saliency`](/saliency.py) quanto o [`GuidedBackprop`](/guided_backprop.py), elevando a qualidade de ambos.

---

## 5. Vantagens e Limitações

### Vantagens:
- **Redução Drástica de Ruído:** Converte mapas granulados e pontilhados em contornos anatômicos muito mais coesos e interpretáveis por radiologistas.
- **Agnóstico ao Método Base:** Pode ser aplicado sobre qualquer método baseado em gradiente diferencial.
- **Atenuação de Mínimos Locais:** Ajuda a escapar de singularidades locais da superfície de decisão da CNN.

### Limitações:
- **Custo Computacional Multiplicado:** Se $N=30$, a geração da explicação levará aproximadamente 30 vezes mais tempo do que a execução simples de um único gradiente.
- **Sensibilidade a Hiperparâmetros:** Se $\sigma$ for muito baixo, o ruído não é cancelado; se for muito alto, características anatômicas importantes são borradas.

---

## 6. Como Usar

```python
import torch
from saliency import Saliency
from guided_backprop import GuidedBackprop
from smoothgrad import SmoothGrad
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo PyTorch carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Opção 1: SmoothGrad com Guided Backpropagation
base_explainer = GuidedBackprop(model, device)
smooth_explainer = SmoothGrad(base_explainer, device, num_samples=30, noise_level=0.15)

result = smooth_explainer.generate(image)
heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="SmoothGrad + Guided BP", show=True)
```
