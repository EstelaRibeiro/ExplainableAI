# RISE: Randomized Input Sampling for Explanation

**Arquivo no projeto:** [`rise.py`](/rise.py)  
**Artigo pioneiro:** *RISE: Randomized Input Sampling for Explanation of Black-box Models* (Petsiuk, Das & Saenko, BMVC 2018 - arXiv:1806.07404)

---

## 1. Intuição Didática

Imagine que você está olhando para uma radiografia através de uma cortina com centenas de furinhos perfurados aleatoriamente: partes da imagem ficam visíveis e partes ficam tapadas. Se você olhar através de uma única cortina com furos aleatórios, a imagem parece confusa.

Porém, se você trocar essa cortina **4.000 vezes por segundo**, cada vez com uma constelação diferente de furos, e em cada uma delas anotar a confiança do médico no diagnóstico de "Aorta Alongada":
- Sempre que os furos revelam a curvatura superior do arco aórtico, a confiança do médico é alta.
- Quando os furos tapam essa região, a confiança despenca.

O **RISE (Randomized Input Sampling for Explanation)** é a formalização estatística dessa ideia:
> *"Em vez de deslizar um quadrado rígido, gere milhares de máscaras aleatórias e suaves, multiplique pela imagem original e pondere cada máscara pela probabilidade que o modelo previu ao visualizá-la."*

Como as máscaras são ampliadas com interpolação bilinear suave a partir de uma grade inicial pequena (ex.: $7 \times 7$), o mapa de calor final resultante é extraordinariamente orgânico, contínuo e sem bordas artificiais.

---

## 2. Fundamentação Matemática

Seja $I$ a imagem de entrada e $f(I) \in [0, 1]$ a probabilidade predita pelo modelo caixa-preta para a classe alvo.

### Passo 1: Geração de Máscaras Suaves
1. Amostra-se uma máscara binária aleatória $m \in \{0, 1\}^{s \times s}$ em uma grade grosseira ($s = 7$), onde cada elemento é uma variável aleatória de Bernoulli independente com probabilidade de sucesso $p$ (`keep_prob = 0.5`):
   $$m_{i, j} \sim \text{Bernoulli}(p)$$
2. A máscara grosseira é redimensionada para a resolução final $(H, W)$ por meio de interpolação bilinear contínua:
   $$M = \text{BilinearUpsample}(m)$$
   Isso produz transições gradientes suaves entre $0$ e $1$.

### Passo 2: Avaliação de Monte Carlo
Para $N$ máscaras independentes $\{M_1, M_2, \dots, M_N\}$ (no projeto, padrão: $N = 4000$), a imagem mascarada $I \odot M_k$ é avaliada pelo modelo:

$$S_k = f(I \odot M_k)$$

### Passo 3: Ponderação e Normalização
O mapa de importância final é a média ponderada de todas as máscaras pelas probabilidades resultantes, normalizada pelo valor esperado da máscara $\mathbb{E}[M] = p$:

$$\text{Heatmap} = \frac{1}{N \cdot p} \sum_{k=1}^{N} S_k \cdot M_k$$

Onde $\text{Heatmap}$ converge assintoticamente para o gradiente de importância esperado sob o processo estocástico.

---

## 3. Fluxo de Execução Passo a Passo

```
1. Geração de Grade Grosseira:
   Amostra N tensores (7 x 7) com Bernoulli(p=0.5).

2. Suavização das Máscaras:
   Interpolação Bilinear: (N, 1, 7, 7) ───> (N, 1, H, W)

3. Aplicação sobre a Imagem:
   Imagens_Mascaradas = Imagem * Máscaras (N imagens no total)

4. Inferência em Lotes (Batching Paralelo):
   Para cada lote de 64 imagens:
   └── Obter probabilidades da rede S_k = Sigmoid(Model(lote))

5. Combinação Linear e Normalização:
   Heatmap = Σ (S_k * M_k) / (N * p)
   Heatmap = normalize(Heatmap)
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`rise.py`](/rise.py):

```python
class RISE:
    def __init__(
        self,
        model,
        device,
        num_masks=4000,
        grid_size=7,
        keep_prob=0.5,
        batch_size=64,
    ):
        self.model = model
        self.device = device
        self.num_masks = num_masks
        self.grid_size = grid_size
        self.keep_prob = keep_prob
        self.batch_size = batch_size

    def _generate_masks(self, image_size):
        H, W = image_size
        # 1. Gera grade aleatória binária
        masks = (
            torch.rand(self.num_masks, 1, self.grid_size, self.grid_size, device=self.device)
            < self.keep_prob
        ).float()

        # 2. Upsampling bilinear produz contornos suaves
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        return masks

    def generate(self, image):
        ...
        masks = self._generate_masks((H, W))
        images = image.expand(self.num_masks, -1, -1, -1)
        masked_images = images * masks

        scores = []
        with torch.no_grad():
            for i in range(0, self.num_masks, self.batch_size):
                batch = masked_images[i:i+self.batch_size]
                batch_scores = self.model(batch)
                scores.append(batch_scores.squeeze())

        scores = torch.cat(scores)

        # 3. Média ponderada analítica do artigo original
        heatmap = (scores.view(-1, 1, 1, 1) * masks).sum(dim=0)
        heatmap /= (self.num_masks * self.keep_prob)
        heatmap = normalize(heatmap.squeeze())
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Zero Pressupostos Internos (Black-Box Total):** Não depende de gradientes, nem de arquitetura convolucional.
- **Transições Extremamente Orgânicas:** As máscaras interpoladas eliminam as quebras quadradas e feias da oclusão tradicional por janela deslizante.
- **Rigor Estatístico:** Baseado na expectativa de Monte Carlo, oferecendo garantias formais de convergência conforme $N \to \infty$.

### Limitações:
- **Custo Computacional Muito Alto:** Requer milhares de passes forward ($N = 4000$). Embora o código use GPU com `batch_size=64`, o processamento leva consideravelmente mais tempo do que métodos baseados em CAM (segundos a minutos por imagem).

---

## 6. Como Usar

```python
import torch
from rise import RISE
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Instanciação (4.000 máscaras avaliadas em lotes de 64)
explainer = RISE(model, device, num_masks=4000, grid_size=7, keep_prob=0.5, batch_size=64)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="RISE", show=True)
```
