# Layer-CAM (LayerGradCAM): Ponderação Espacial Ponto a Ponto

**Arquivo no projeto:** [`layergradcam.py`](/layergradcam.py)  
**Artigo pioneiro:** *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization* (Jiang, Zhang, Hou, Cheng & Wei, IEEE Transactions on Image Processing - TIP 2021 - arXiv:2102.08983)

---

## 1. Intuição Didática

Nos métodos Grad-CAM e Grad-CAM++, a importância de um canal convolucional é sempre resumida em um **único número escalar** por meio de uma média global (Global Average Pooling). No entanto, um mesmo canal convolucional pode conter ativações em diferentes cantos da imagem: um ponto no canto superior pode estar detectando a borda da aorta (altamente relevante), enquanto outro ponto no canto inferior pode ser apenas ruído ou tecido sem importância.

Ao aplicar uma média global sobre o canal, a relevância do ponto importante é "diluída" e o ruído do ponto inútil ganha peso.

O **Layer-CAM** (no projeto chamado `LayerGradCAM`) quebra essa limitação com uma ideia brilhante e simples:
> *"Em vez de resumir o gradiente de um canal em um único número médio para toda a imagem, mantenha a matriz espacial de gradientes intacta! Pondere cada pixel da ativação exatamente pelo gradiente daquele mesmo pixel."*

Se um pixel específico do mapa de ativação tiver gradiente positivo alto, ele ganha peso máximo. Se outro pixel do mesmo canal não tiver gradiente, ele é ignorado. O resultado é um mapa muito mais refinado, nítido e com menor espalhamento espacial.

---

## 2. Fundamentação Matemática

Seja $A_{i, j}^k$ o valor de ativação do canal $k$ na coordenada espacial $(i, j)$, e $\frac{\partial Y^c}{\partial A_{i, j}^k}$ o respectivo gradiente.

### Ponderação Espacial Elemento a Elemento:
Ao contrário do Grad-CAM que faz a média $\sum_{i,j}$, o Layer-CAM calcula o peso espacialmente distribuído:

$$w_{i, j}^{kc} = \text{ReLU}\left( \frac{\partial Y^c}{\partial A_{i, j}^k} \right)$$

Observe que $w_{i, j}^{kc} \in \mathbb{R}^{H' \times W'}$ é uma **matriz bidimensional** do mesmo tamanho da ativação, e não um escalar isolado.

### Combinação Espacial com Hadamard (Produto Ponto a Ponto):
O mapa de ativação é obtido pelo produto elemento a elemento e somatório nos canais $k$:

$$L_{\text{LayerCAM}}^c = \text{ReLU}\left( \sum_{k} w_{i, j}^{kc} \cdot A_{i, j}^k \right)$$

Após a combinação, o mapa passa por $\text{ReLU}$ final, interpolação bilinear para a dimensão de entrada da imagem e normalização $[0, 1]$.

---

## 3. Comparativo Estrutural de Ponderação

```
GRAD-CAM:
Ativação A^k (H x W)  ✕  Gradiente Médio Escalar α^k  ───>  Mancha difusa uniforme

LAYER-CAM:
Ativação A^k (H x W)  ⊙  Gradiente Espacial ReLU(G^k) (H x W)  ───>  Ativação seletiva refinada
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`layergradcam.py`](/layergradcam.py):

```python
class LayerGradCAM:
    ...
    def generate(self, image):
        ...
        score = output.squeeze()
        score.backward()

        activations = self.activations # Shape: (1, C, H', W')
        gradients = self.gradients     # Shape: (1, C, H', W')

        # ----------------------------------------------------
        # AQUI A DIFERENÇA CRUCIAL ENTRE GRADCAM E LAYERCAM:
        # Não existe .mean(dim=(2, 3))!
        # Mantém a resolução espacial (H', W') aplicando ReLU direto.
        # ----------------------------------------------------
        weights = F.relu(gradients)

        # Multiplicação ponderada ponto a ponto elemento a elemento
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Interpolação bilinear para resolução da imagem (384, 384)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze()
        cam = normalize(cam)
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Resolução e Foco Anatômico Notáveis:** Elimina os halos e "vazamentos" de calor para fora da estrutura anatômica de interesse.
- **Hierarquia de Camadas:** Pode ser aplicado tanto em camadas profundas quanto em camadas convolucionais intermediárias sem colapso visual (ao contrário do Grad-CAM tradicional que gera ruído em camadas intermediárias).
- **Sem Sobrecarga de Tempo:** Tão rápido quanto o Grad-CAM tradicional (1 forward + 1 backward), pois o produto matricial no PyTorch é totalmente paralelizado na GPU.

### Limitações:
- **Sensibilidade a Zeros:** Como aplica $\text{ReLU}$ tanto nos gradientes quanto no mapa final, qualquer ponto com gradiente ligeiramente nulo ou negativo é prontamente suprimido, o que pode fragmentar áreas patológicas muito extensas.

---

## 6. Como Usar

```python
import torch
from layergradcam import LayerGradCAM
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

explainer = LayerGradCAM(model, device)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="LayerCAM", show=True)
```
