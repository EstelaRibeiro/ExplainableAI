# Vanilla Saliency Maps

**Arquivo no projeto:** [`saliency.py`](/saliency.py)  
**Artigo pioneiro:** *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps* (Simonyan, Vedaldi & Zisserman, 2013 - arXiv:1312.6034)

---

## 1. Intuição Didática

Imagine que você está diante de uma mesa de som com centenas de botões giratórios, e a saída de áudio é o volume de uma música. Se você quiser descobrir quais botões têm maior impacto imediato sobre o volume, você dá um "toquezinho" sutil em cada botão para a direita ou esquerda e observa o quanto o volume sobe ou desce.

O **Vanilla Saliency Map** faz exatamente isso com os pixels de uma imagem:
> *"Se eu alterar sutilmente o valor de brilho deste pixel específico, o quanto a pontuação da classe (ex.: Aorta Alongada) irá variar?"*

Pixels onde uma alteração infinitesimal provoca uma variação brusca na predição do modelo recebem alta saliência (relevância). Pixels que, se alterados, não mudam a resposta da rede recebem saliência zero.

---

## 2. Formulação Matemática

Dada uma imagem de entrada $I \in \mathbb{R}^{C \times H \times W}$ e uma função de pontuação do modelo $S_c(I)$ (o *logit* não-normalizado para a classe $c$), a aproximação linear de Taylor de primeira ordem da função em torno de $I$ é dada por:

$$S_c(I) \approx w^T I + b$$

Onde o vetor de pesos $w$ é exatamente a derivada parcial da saída em relação a cada elemento da entrada:

$$w = \left. \frac{\partial S_c}{\partial I} \right|_{I}$$

Como uma imagem colorida possui canais RGB ($C=3$), o mapa de saliência bidimensional $M(x, y)$ é tipicamente obtido tomando a magnitude máxima absoluta entre os canais de cor para cada posição espacial $(x, y)$:

$$M(x, y) = \max_{c \in \{R, G, B\}} |w(c, x, y)|$$

Em seguida, o mapa é normalizado para o intervalo $[0, 1]$ para exibição como mapa de calor:

$$M_{\text{norm}}(x, y) = \frac{M(x, y) - \min(M)}{\max(M) - \min(M) + \epsilon}$$

---

## 3. Fluxo de Execução Passo a Passo

```
[ Imagem de Entrada I ]
          │
          ▼
[ 1. Habilitar requires_grad_(True) na Imagem ]
          │
          ▼
[ 2. Forward Pass: Obter logit de predição S_c ]
          │
          ▼
[ 3. Backward Pass: S_c.backward() ]
          │
          ▼
[ 4. Extrair Imagem.grad: ∂S_c / ∂I ]
          │
          ▼
[ 5. Reduzir canais: max(|grad|) ao longo da dimensão C ]
          │
          ▼
[ 6. Normalizar [0, 1] -> Heatmap ]
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`saliency.py`](/saliency.py):

```python
class Saliency:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate(self, image):
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)

        # Habilita o rastreamento de gradiente no tensor da imagem
        image.requires_grad_(True)

        if image.grad is not None:
            image.grad.zero_()
        self.model.zero_grad()

        # Forward pass
        output = self.model(image)
        
        # Backward pass calcula d(output)/d(image)
        output.backward()

        gradient = image.grad.detach()

        # Pega a maior magnitude entre os 3 canais de cor (C=1)
        heatmap, _ = gradient.abs().max(dim=1)
        heatmap = heatmap.squeeze()

        # Normaliza entre 0 e 1 usando utils_xai.normalize
        heatmap = normalize(heatmap)

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.detach().cpu(),
            "probability": torch.sigmoid(output).detach().cpu(),
            "prediction": (torch.sigmoid(output) >= 0.5).int().cpu()
        }
```

### Destaques do Código:
- **`image.requires_grad_(True)`**: Normalmente tensores de entrada não armazenam gradiente. É fundamental ativar essa flag antes do forward pass.
- **`output.backward()`**: Retropropaga o sinal da classe binária diretamente para a imagem.
- **`gradient.abs().max(dim=1)`**: Reduz o tensor de formato `(1, 3, H, W)` para `(1, H, W)`, retendo a maior perturbação nos canais RGB.

---

## 5. Vantagens e Limitações

### Vantagens:
- **Resolução Nativa:** Cada pixel da imagem de entrada possui sua própria pontuação de atribuição.
- **Ultra Rápido:** Requer apenas um único passe forward e um único passe backward.
- **Fundamento Matemático Elegante:** Derivação direta do cálculo diferencial de primeira ordem.

### Limitações:
- **Ruído Visual Severo (Salt-and-Pepper):** Devido a variações locais bruscas e saturação não linear nas ativações da rede (como ReLUs e MaxPoolings), os mapas de saliência pura costumam parecer "ruído de TV fora do ar".
- **Saturação de Gradientes:** Se uma característica já saturou a ativação interna do neurônio, sua derivada local pode ser zero mesmo sendo crucial para a decisão.
- **Pouca Clareza Anatômica:** Em radiografias, dificilmente forma áreas coesas de órgãos, destacando apenas bordas isoladas.

---

## 6. Como Usar

```python
import torch
from saliency import Saliency
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo PyTorch em modo eval

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

explainer = Saliency(model, device)
result = explainer.generate(image)

heatmap = result["heatmap"]
print(f"Probabilidade predita: {result['probability'].item():.4f}")

# Visualizar sobreposição
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Vanilla Saliency", show=True)
```
