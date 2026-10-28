# Grad-CAM (Gradient-weighted Class Activation Mapping)

**Arquivo no projeto:** [`gradcam.py`](/gradcam.py)  
**Artigo pioneiro:** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization* (Selvaraju, Cogswell, Das, Vedaldi, Parikh & Batra, ICCV 2017 - arXiv:1610.02391)

---

## 1. Intuição Didática

Imagine que você está assistindo a um filme através de um projetor que possui várias lentes sobrepostas, e cada lente foca em um aspecto diferente da cena: uma lente realça contornos anatômicos, outra destaca texturas de vasos sanguíneos, outra foca em densidades de ar no pulmão. No final, o cérebro do diretor (o classificador) decide se aquela imagem mostra uma "Aorta Alongada".

O **Grad-CAM** faz duas perguntas cruciais à rede convolucional:
1. **Quais lentes (mapas de ativação da última camada convolucional) estavam prestando atenção em cada pedaço da imagem?**
2. **Quanto o classificador final se importou com o que cada lente específica viu?**

O método calcula a importância de cada mapa de ativação tirando a média dos gradientes associados a ele. Em seguida, soma todas as lentes ponderadas por suas respectivas importâncias e aplica um filtro **ReLU** para descartar tudo o que não contribuiu positivamente para a classe.

O resultado é uma "mancha de calor" anatômica ampla, estável e intuitiva que aponta a região macroscópica que motivou a decisão.

---

## 2. Fundamentação Matemática

Seja $A^k \in \mathbb{R}^{H' \times W'}$ o $k$-ésimo mapa de ativação gerado pela última camada convolucional da rede, e $y^c$ a pontuação não normalizada (logit) para a classe alvo $c$.

### Passo 1: Cálculo dos Pesos dos Canais ($\alpha_k^c$)
O peso de importância do canal $k$ para a classe $c$ é obtido aplicando o **Global Average Pooling (GAP)** sobre as derivadas parciais:

$$\alpha_k^c = \overbrace{\frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'}}^{\text{Média Global (GAP)}} \frac{\partial y^c}{\partial A_{i, j}^k}$$

Onde $Z = H' \times W'$ é o número total de elementos espaciais na grade do mapa de ativação.

### Passo 2: Combinação Linear Ponderada e ReLU
O mapa final de localização $L_{\text{Grad-CAM}}^c$ é construído multiplicando cada mapa de ativação $A^k$ pelo seu respectivo peso $\alpha_k^c$, somando todos os canais e aplicando uma função de ativação retificadora ($\text{ReLU}$):

$$L_{\text{Grad-CAM}}^c = \text{ReLU} \left( \sum_{k} \alpha_k^c A^k \right)$$

> **Por que aplicar ReLU?**  
> A função $\text{ReLU}$ garante que o mapa retenha apenas características que possuem correlação **positiva** com a classe de interesse (isto é, aumentam $y^c$). Características que reduzem a confiança na classe pertencem a outras categorias e são zeradas.

### Passo 3: Interpolação Bilinear
Como a última camada convolucional tem dimensão reduzida (ex.: $12 \times 12$ numa ResNet com entrada $384 \times 384$), o mapa $L^c$ é ampliado via interpolação bilinear para casar com a resolução original da imagem e depois normalizado para o intervalo $[0, 1]$.

---

## 3. Fluxo de Execução Passo a Passo

```
[ Imagem de Entrada x ] 
          │
          ▼
[ Forward Pass até a Última Camada Convolucional ] 
          │───> Capturar Ativações A^k via Forward Hook
          ▼
[ Classificador / Head: Obter Logit y^c ]
          │
          ▼
[ Backward Pass: y^c.backward() ] 
          │───> Capturar Gradientes ∂y^c / ∂A^k via Backward Hook
          ▼
[ Global Average Pooling dos Gradientes: pesos α_k^c ]
          │
          ▼
[ Combinação Linear: Σ (α_k^c * A^k) ]
          │
          ▼
[ ReLU: descarta valores negativos ]
          │
          ▼
[ Interpolação Bilinear para resolução original H x W ]
          │
          ▼
[ Normalização Min-Max [0, 1] ]
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`gradcam.py`](/gradcam.py):

```python
class GradCAM:
    def __init__(self, model, device, target_layer=None):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self.target_layer = target_layer

    def _find_target_layer(self, image):
        """Descobre automaticamente a última camada Conv2d 4D"""
        last_layer = None
        def hook(module, input, output):
            nonlocal last_layer
            if (isinstance(module, torch.nn.Conv2d) and 
                output.ndim == 4 and output.shape[-2] > 1):
                last_layer = module
        ...

    def _register_hooks(self):
        """Registra hooks no PyTorch para interceptar tensores internos"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image):
        ...
        score = output.squeeze()
        score.backward()

        # 1. Global Average Pooling (média espacial dos gradientes)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 2. Ponderação dos mapas de ativação
        cam = (weights * self.activations).sum(dim=1)

        # 3. Retificação ReLU
        cam = F.relu(cam)

        # 4. Upsampling bilinear para (384, 384)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = normalize(cam.squeeze())
```

### Destaque da Implementação:
A função `_find_target_layer` elimina a necessidade de saber o nome exato da camada interna (`model.layer4[2].conv3`), inspecionando dinamicamente a topologia do modelo em tempo de execução.

---

## 5. Vantagens e Limitações

### Vantagens:
- **Alta Legibilidade Semântica:** Como atua na camada mais profunda da CNN, identifica conceitos de alto nível (órgãos inteiros, lesões focais).
- **Robusto a Ruído:** Não sofre com a granulação dos métodos de pixel puro.
- **Rápido:** Exige apenas 1 passe forward e 1 passe backward.
- **Discriminação de Classe:** Se o modelo tiver múltiplas classes, o mapa aponta com clareza para a classe selecionada.

### Limitações:
- **Resolução Grosseira:** Devido ao subamostragem sucessiva (*pooling* e *strides* das convoluções), a grade original pode ser de apenas $12 \times 12$. A interpolação bilinear gera uma "mancha borrada" sem contornos nítidos das bordas anatômicas.
- **Ponderação Média Subótima (GAP):** Em casos onde há múltiplos alvos na mesma imagem ou objetos com tamanhos variados, a média global pode diluir a importância de regiões menores. (Motivação para o surgimento do Grad-CAM++).

---

## 6. Como Usar

```python
import torch
from gradcam import GradCAM
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo PyTorch carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Instanciação (descobre automaticamente a última camada convolucional)
explainer = GradCAM(model, device)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Grad-CAM", show=True)
```
