# Integrated Gradients (IG)

**Arquivo no projeto:** [`ig.py`](i/ig.py)  
**Artigo pioneiro:** *Axiomatic Attribution for Deep Networks* (Sundararajan, Taly & Yan, Google, ICML 2017 - arXiv:1703.01365)

---

## 1. Intuição Didática

Imagine que você está em uma sala completamente escura e acende gradualmente um regulador de luz (*dimmer*) do nível 0% (escuridão total) até 100% (luz plena). Conforme a luz sobe de intensidade, você observa em que momento específico cada objeto da sala (um quadro na parede, um vaso no chão) se torna visível e faz você reconhecer o ambiente.

O método **Integrated Gradients** opera exatamente sob essa premissa:
> *"Em vez de calcular o gradiente apenas na imagem final saturada, pegue uma imagem de referência neutra ('baseline', como uma imagem preta sem informação) e crie uma sequência de imagens intermediárias que aumentam o brilho gradualmente até a imagem real. Calcule o gradiente em cada etapa desse caminho e integre (some) todos eles."*

Isso resolve o clássico problema da **saturação de gradientes**: quando um neurônio já está com ativação máxima, a derivada local do gradiente se torna quase zero, fazendo com que o método Vanilla Saliency ignore pixels cruciais. A integração ao longo do caminho restaura a importância real desses pixels.

---

## 2. Fundamentação Axiomática e Matemática

Integrated Gradients foi desenhado para satisfazer dois axiomas matemáticos fundamentais da teoria de atribuição:

1. **Axioma da Completude (*Completeness*):** A soma das atribuições de todos os pixels deve ser rigorosamente igual à diferença entre a pontuação do modelo para a imagem original $F(x)$ e a pontuação para a imagem de referência neutra $F(x')$:
   $$\sum_i \text{IG}_i(x) = F(x) - F(x')$$
2. **Invariância de Implementação (*Implementation Invariance*):** Dois modelos funcionalmente idênticos (que produzem as mesmas saídas para as mesmas entradas) devem receber exatamente as mesmas explicações, independente de sua arquitetura interna.

### Formulação Integral:
Para a $i$-ésima dimensão (pixel) da imagem de entrada $x$, em relação à linha de base $x'$:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

### Aproximação de Riemann por Soma Discreta:
Na prática computacional, a integral contínua é calculada aproximando-a pela soma dos gradientes em $m$ passos discretos igualmente espaçados (no projeto, padrão: $m = 50$):

$$\text{IG}_i^{\text{aprox}}(x) = (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

---

## 3. Fluxo de Execução Passo a Passo

```
1. Definir Linha de Base (Baseline x'):
   Geralmente um tensor de zeros (imagem preta: x' = 0).

2. Calcular Vetor de Diferença:
   delta = x - x'

3. Interpolação Linear (m passos de alpha de 0 a 1):
   Para alpha in [1/m, 2/m, ..., 1.0]:
   ├── Imagem_interpolar = x' + alpha * delta
   ├── Forward Pass no modelo
   ├── Backward Pass: obter gradiente em relação à imagem interpolada
   └── Acumular gradiente na soma total

4. Gradiente Médio:
   gradiente_medio = gradiente_acumulado / m

5. Atribuição Final:
   atribuicao = delta * gradiente_medio

6. Mapa de Calor (Heatmap):
   heatmap = max(|atribuicao|, canais_RGB)
   heatmap = normalize(heatmap)
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`ig.py`](/ig.py):

```python
class IntegratedGradients:
    def __init__(self, model, device, steps=50):
        self.model = model
        self.device = device
        self.steps = steps

    def generate(self, image, baseline=None):
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)

        # 1. Baseline padrão: tensor de zeros (imagem preta)
        if baseline is None:
            baseline = torch.zeros_like(image)
        else:
            baseline = baseline.unsqueeze(0).to(self.device)

        delta = image - baseline
        accumulated_gradients = torch.zeros_like(image)

        # 2. Coeficientes de interpolação alpha de 0 a 1
        alphas = torch.linspace(0, 1, self.steps, device=self.device)

        for alpha in alphas:
            interpolated = baseline + alpha * delta
            interpolated.requires_grad_(True)

            self.model.zero_grad()
            output = self.model(interpolated)
            score = output.squeeze()
            score.backward()

            accumulated_gradients += interpolated.grad.detach()

        # 3. Média dos gradientes e multiplicação pelo delta
        avg_gradients = accumulated_gradients / self.steps
        integrated_gradients = delta * avg_gradients

        # 4. Redução RGB -> Grayscale e normalização
        heatmap, _ = integrated_gradients.abs().max(dim=1)
        heatmap = normalize(heatmap.squeeze())

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.detach().cpu(),
            "probability": torch.sigmoid(output).detach().cpu(),
            "prediction": (torch.sigmoid(output) >= 0.5).int().cpu()
        }
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Solidez Teórica Inigualável:** É um dos poucos métodos com respaldo formal em teoria dos jogos (extensão contínua dos valores de Shapley) e axiomas de garantia matemática.
- **Não Satura:** Recupera a importância de regiões que o Vanilla Saliency zera indevidamente por causa da saturação das funções de ativação.
- **Fidelidade Causal:** O produto $(x - x') \times \nabla_{\text{médio}}$ garante que se um pixel for idêntico à linha de base ($x_i = x'_i$), sua atribuição será rigorosamente zero.

### Limitações:
- **Escolha da Baseline:** Em imagens médicas com normalização ImageNet (onde a média já é subtraída), uma baseline de zeros absolutos pode não representar um "vazio sem informação" clinicamente neutro, influenciando o heatmap.
- **Custo Computacional:** Requer `steps` passes forward e backward (no caso, 50 vezes mais lento que Saliency simples).

---

## 6. Como Usar

```python
import torch
from ig import IntegratedGradients
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo PyTorch carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

explainer = IntegratedGradients(model, device, steps=50)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar sobreposição
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Integrated Gradients", show=True)
```
