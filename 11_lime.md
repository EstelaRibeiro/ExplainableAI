# LIME: Local Interpretable Model-agnostic Explanations

**Arquivo no projeto:** [`lime.py`](/lime.py)  
**Artigo pioneiro:** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (Ribeiro, Singh & Guestrin, ACM KDD 2016 - arXiv:1602.04938)

---

## 1. Intuição Didática

Imagine a superfície do planeta Terra: ela é redonda e curva em escala global. No entanto, quando você está pisando no chão da sua sala de estar, para todos os efeitos práticos do seu dia a dia, o chão parece perfeitamente plano!

O **LIME** aplica esse mesmo princípio fundamental aos modelos de Inteligência Artificial:
> *"Uma Rede Neural Convolucional profunda possui uma fronteira de decisão global incrivelmente complexa, cheia de curvas e não-linearidades. Mas se dermos um 'zoom' e olharmos apenas para os arredores imediatos de UMA imagem específica, a decisão da rede se comporta de forma aproximadamente plana (linear)!"*

Em imagens, o LIME não avalia pixels soltos (que isolados não possuem significado humano). Em vez disso, ele agrupa pixels vizinhos com textura e brilho semelhantes em **Superpixels** (regiões anatômicas coesas, como o arco aórtico, a silhueta cardíaca ou um arco costal). Depois, "liga e desliga" esses superpixels aleatoriamente e treina um modelo linear simples para descobrir quais regiões empurram a predição para cima ou para baixo.

---

## 2. Fundamentação Matemática

O objetivo do LIME é minimizar a seguinte função de perda:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Onde:
- $f(x)$ é o modelo caixa-preta complexo original (a CNN).
- $g \in G$ é o modelo explicador interpretável simples (uma Regressão Linear com penalidade Ridge).
- $\Omega(g)$ é a medida de complexidade do explicador (controlada pelo número de superpixels).
- $\pi_x(z)$ é a função kernel de proximidade (peso de localidade), que dá mais valor às amostras muito parecidas com a imagem original e menos valor às amostras descaracterizadas:

$$\pi_x(z) = \exp \left( - \frac{D_{\text{cosseno}}(z, \mathbf{1})^2}{\sigma^2} \right)$$

Onde $D_{\text{cosseno}}$ é a distância de cosseno entre a máscara perturbada $z \in \{0, 1\}^S$ e a imagem original $\mathbf{1}$, e $\sigma$ é a largura do kernel (no projeto: $\sigma = 0.25$).

O modelo substituto (*surrogate*) ajustado via **Ridge Regression** resolve:

$$\min_{w} \sum_{k=1}^{N} \pi_x(z_k) \cdot \left( f(x_k) - w^T z_k \right)^2 + \alpha \|w\|_2^2$$

Os coeficientes $w = [w_1, w_2, \dots, w_S]$ representam a importância linear direta de cada um dos $S$ superpixels.

---

## 3. Fluxo de Execução Passo a Passo

```
[ 1. Imagem Original ]
          │
          ▼
[ 2. Segmentação em Superpixels via SLIC ]
     Divide a radiografia em ~150 regiões anatômicas contíguas
          │
          ▼
[ 3. Geração de Amostras Perturbadas (N = 1000) ]
     Cria matriz binária (1000 x 150): 1 = mantém superpixel; 0 = substitui pela cor média
          │
          ▼
[ 4. Avaliação na CNN (Forward em Lotes de 64) ]
     Obtém as probabilidades previstas f(z_k) para cada perturbação
          │
          ▼
[ 5. Cálculo dos Pesos de Proximidade π_x(z) ]
     Calcula a distância cosseno em relação à imagem original
          │
          ▼
[ 6. Ajuste do Modelo Substituto (Ridge Regression) ]
     Ajusta o modelo linear ponderado para extrair os coeficientes w
          │
          ▼
[ 7. Construção do Mapa de Calor (Heatmap) ]
     Pinta cada superpixel da imagem com o seu respectivo coeficiente w_s
     Normaliza no intervalo [0, 1]
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`lime.py`](/lime.py):

```python
class LIME:
    def __init__(self, model, device, num_samples=1000, num_segments=150, compactness=10, sigma=0.25, batch_size=64):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.compactness = compactness
        self.sigma = sigma
        self.batch_size = batch_size
        ...

    def _segment_image(self, image_original):
        """Segmentação usando SLIC (Simple Linear Iterative Clustering)"""
        image_np = image_original.permute(1, 2, 0).cpu().numpy()
        return slic(image_np, n_segments=self.num_segments, compactness=self.compactness, start_label=0)

    def _generate_masks(self, num_superpixels):
        """1 = mantém superpixel; 0 = oculta"""
        masks = np.random.randint(0, 2, size=(self.num_samples, num_superpixels), dtype=np.uint8)
        masks[0] = 1 # O primeiro exemplo é a imagem original sem modificações
        return masks

    def _build_batch(self, image_original, segments, batch_masks):
        """Substitui superpixels desativados pela cor média da imagem"""
        ...
        mean_color = image_original.mean(dim=(1, 2))
        for sp in range(batch_masks.shape[1]):
            region = segments == sp
            remove = batch_masks[:, sp] == 0
            if remove.any():
                idx = np.where(remove)[0]
                for i in idx:
                    images[i][:, region] = mean_color.unsqueeze(1)
        ...

    def _compute_weights(self, superpixel_masks):
        """Kernel exponencial sobre a distância cosseno"""
        original = np.ones((1, superpixel_masks.shape[1]))
        distances = cosine_distances(superpixel_masks, original).ravel()
        return np.exp(-(distances ** 2) / (self.sigma ** 2))

    def _fit_surrogate(self, superpixel_masks, scores, weights):
        """Ajusta o modelo linear Ridge"""
        model = Ridge(alpha=1.0)
        model.fit(superpixel_masks, scores, sample_weight=weights)
        return model.coef_

    def _build_heatmap(self, segments, coefficients):
        """Mapeia os coeficientes de volta aos superpixels"""
        heatmap = np.zeros_like(segments, dtype=np.float32)
        for sp, coef in enumerate(coefficients):
            heatmap[segments == sp] = coef
        return normalize(torch.from_numpy(heatmap))
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Interpretabilidade Anatômica Altíssima:** Como os superpixels respeitam as bordas reais dos órgãos e tecidos (via algoritmo SLIC), as explicações não parecem borrões arbitrários, mas sim recortes de estruturas anatômicas inteiras.
- **Modelo Substituto Transparente:** Os pesos da regressão linear indicam com precisão quantitativa quanto cada segmento contribuiu positivamente ou negativamente para a decisão.
- **Agnóstico ao Modelo:** Funciona com qualquer classificador.

### Limitações:
- **Dependência da Qualidade do SLIC:** Se a segmentação inicial agrupar dois tecidos anatômicos diferentes no mesmo superpixel, o LIME não conseguirá distingui-los.
- **Instabilidade Estocástica:** Como as perturbações são geradas aleatoriamente, duas execuções sucessivas do LIME na mesma imagem podem produzir pequenas variações visuais.
- **Custo Computacional:** Requer `num_samples` (1.000) avaliações na rede neural.

---

## 6. Como Usar

```python
import torch
from lime import LIME
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image_norm = sample["normalized"]
image_orig = sample["original"]

# No LIME, passamos o tensor normalizado e a imagem original para o SLIC
explainer = LIME(model, device, num_samples=1000, num_segments=150)
result = explainer.generate(image=image_norm, image_original=image_orig)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="LIME", show=True)
```
