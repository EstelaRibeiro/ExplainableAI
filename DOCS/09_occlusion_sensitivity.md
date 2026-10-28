# Occlusion Sensitivity: Teste de Perturbação por Oclusão Deslizante

**Arquivo no projeto:** [`occlusion.py`](/occlusion.py)  
**Artigo pioneiro:** *Visualizing and Understanding Convolutional Networks* (Zeiler & Fergus, ECCV 2014 - arXiv:1311.2901)

---

## 1. Intuição Didática

Imagine que você está fazendo um teste oftalmológico ou montando um quebra-cabeça. Para descobrir qual elemento de uma foto é essencial para você reconhecê-la como um "carro", você pega um pedaço quadrado de papelão cinza e tapa as rodas: você ainda consegue reconhecer o carro. Agora você tapa o farol: ainda reconhece. Mas se você tapar o corpo principal do veículo, de repente você não consegue mais identificá-lo com certeza.

O método de **Sensibilidade por Oclusão (Occlusion Sensitivity)** aplica rigorosamente esse teste de "tapar e observar":
> *"Pegue uma pequena janela cinza (ex.: $32 \times 32$ pixels) e deslize-a por toda a radiografia com passos regulares. A cada posição em que a janela tapa um pedaço da imagem, calcule a predição da rede. Se ao tampar aquele pedaço específico a probabilidade de 'Aorta Alongada' despencar, significa que aquela região era indispensável para o diagnóstico."*

---

## 2. Fundamentação Matemática

Seja $I$ a imagem original e $f(I)$ o logit predito pelo modelo para a classe de interesse.

### Janela Deslizante e Oclusão:
Para uma janela de tamanho $W_{\text{size}} \times W_{\text{size}}$ posicionada na coordenada $(y, x)$, construímos a imagem perturbada $I_{(y, x)}$ substituindo a região retangular por um valor constante neutro $v_{\text{occ}}$ (no projeto, $0.0$):

$$I_{(y, x)}[i, j] = \begin{cases} v_{\text{occ}}, & \text{se } y \le i < y + W_{\text{size}} \text{ e } x \le j < x + W_{\text{size}} \\ I[i, j], & \text{caso contrário} \end{cases}$$

### Cálculo da Importância:
A importância da região ocluída é dada pela **queda na pontuação da classe** em relação à predição original:

$$\Delta S(y, x) = f(I) - f(I_{(y, x)})$$

- Se $\Delta S(y, x) > 0$: A região era essencial para sustentar a predição positiva da classe (a pontuação caiu quando foi coberta).
- Se $\Delta S(y, x) < 0$: A região estava atrapalhando ou competindo contra a classe.

### Acumulação e Média por Sobreposição:
Como a janela se move com um passo menor que o tamanho da janela ($\text{stride} < W_{\text{size}}$), os blocos se sobrepõem. Cada pixel $(i, j)$ acumula a importância de todas as janelas que o cobriram:

$$H(i, j) = \frac{\sum_{(y, x) \ni (i, j)} \Delta S(y, x)}{\max(1, \text{contagem}(i, j))}$$

Em seguida, o mapa $H$ é normalizado para o intervalo $[0, 1]$.

---

## 3. Fluxo de Execução Passo a Passo

```
1. Predição Original:
   Calcular original_score = Model(Imagem)

2. Inicializar Matrizes:
   heatmap = zeros(H, W)
   count_map = zeros(H, W)

3. Varredura com Janela Deslizante (y = 0 até H, x = 0 até W com passo stride):
   ├── Criar cópia da imagem: occluded = image.clone()
   ├── Substituir bloco [y:y+window, x:x+window] por 0.0
   ├── Forward Pass: score = Model(occluded)
   ├── Calcular importancia = original_score - score
   ├── Somar: heatmap[bloco] += importancia
   └── Somar: count_map[bloco] += 1

4. Média Ponderada:
   heatmap = heatmap / count_map

5. Normalização Min-Max [0, 1]
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`occlusion.py`](/occlusion.py):

```python
class Occlusion:
    def __init__(self, model, device, window_size=32, stride=16, occlusion_value=0.0):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.occlusion_value = occlusion_value

    def generate(self, image):
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)

        # 1. Pontuação original da imagem intacta
        with torch.no_grad():
            output = self.model(image)
        original_score = output.squeeze()

        _, _, H, W = image.shape
        heatmap = torch.zeros((H, W), device=self.device)
        count_map = torch.zeros((H, W), device=self.device)

        # 2. Laços da janela deslizante
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y2 = min(y + self.window_size, H)
                x2 = min(x + self.window_size, W)

                occluded = image.clone()
                occluded[:, :, y:y2, x:x2] = self.occlusion_value

                with torch.no_grad():
                    score = self.model(occluded).squeeze()

                # 3. Diferença de confiança
                importance = original_score - score

                heatmap[y:y2, x:x2] += importance
                count_map[y:y2, x:x2] += 1

        # 4. Normalização pelo número de sobreposições
        heatmap = heatmap / count_map.clamp(min=1)
        heatmap = normalize(heatmap)
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Modelo-Agnóstico e Intuitivo:** Não requer acesso a gradientes, camadas internas ou funções de ativação. Funciona com qualquer rede (CNNs, Vision Transformers, etc.).
- **Genuinamente Causal:** Testa intervenções físicas diretas no dado de entrada.
- **Resolução Ajustável:** O usuário tem controle direto sobre a granularidade ajustando `window_size` e `stride`.

### Limitações:
- **Custo Computacional Alto:** O número de passagens para a frente é proporcional a $\left(\frac{H}{\text{stride}}\right) \times \left(\frac{W}{\text{stride}}\right)$. Para uma imagem de $384 \times 384$ com stride 16, são necessárias cerca de $24 \times 24 = 576$ inferências.
- **Artefatos de Borda Quadrada:** Como as janelas são blocos retangulares, o mapa de calor pode exibir pequenas descontinuidades angulares em forma de grade.

---

## 6. Como Usar

```python
import torch
from occlusion import Occlusion
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Instanciação (janela de 32x32 com passo de 16 pixels)
explainer = Occlusion(model, device, window_size=32, stride=16, occlusion_value=0.0)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Occlusion Sensitivity", show=True)
```
