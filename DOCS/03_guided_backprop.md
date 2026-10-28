# Guided Backpropagation (Guided BP)

**Arquivo no projeto:** [`guided_backprop.py`](/guided_backprop.py)  
**Artigo pioneiro:** *Striving for Simplicity: The All Convolutional Net* (Springenberg, Dosovitskiy, Brox & Riedmiller, ICLR 2015 - arXiv:1412.6806)

---

## 1. Intuição Didática

Quando uma Rede Neural Convolucional processa uma imagem médica, cada neurônio intermediário aprende a detectar padrões: bordas suaves, densidades ósseas, silhuetas de vasos ou contornos de tecidos moles. Na retropropagação padrão (Backpropagation), gradientes positivos e negativos fluem livremente para trás. Gradientes negativos agem como um sinal de "isto diminui a probabilidade".

No entanto, quando queremos **visualizar** quais estruturas anatômicas ativaram o neurônio de interesse, gradientes negativos introduzem ruído destrutivo e contrapressão visual.

O **Guided Backpropagation** combina o melhor de dois mundos:
1. O **Backpropagation Padrão** (bloqueia o gradiente onde a ativação para a frente foi negativa, ou seja, $x \le 0$).
2. O **Deconvolutional Network (DeconvNet de Zeiler-Fergus)** (bloqueia gradientes negativos vindos de camadas superiores, ou seja, $grad \le 0$).

> *"Deixe passar apenas o sinal positivo que constrói a evidência: só retropropague o gradiente se o neurônio tiver disparado para a frente ($x > 0$) E se o sinal vindo de trás estiver indicando um aumento da predição ($grad > 0$)."*

O resultado é um desenho nítido, detalhado e semelhante a um croqui a lápis das bordas e texturas anatômicas que mais contribuíram para o modelo.

---

## 2. Comparativo de Regras de Retropropagação

A grande sacada do método reside na função de ativação **ReLU** ($f(x) = \max(0, x)$). Seja $f_i^l$ a ativação da camada $l$ e $R_i^{l+1}$ o gradiente retropropagado da camada superior:

| Técnica | Regra de Retropropagação no ReLU | Significado Didático |
| :--- | :---: | :--- |
| **Standard Backprop** | $R_i^l = (f_i^l > 0) \cdot R_i^{l+1}$ | Bloqueia se a entrada original era $\le 0$. |
| **DeconvNet** | $R_i^l = (R_i^{l+1} > 0) \cdot R_i^{l+1}$ | Bloqueia se o gradiente que chega é $\le 0$. |
| **Guided Backprop** | $R_i^l = (f_i^l > 0) \cdot (R_i^{l+1} > 0) \cdot R_i^{l+1}$ | **Filtro duplo:** só passa se ambos forem positivos! |

---

## 3. Fluxo de Execução Passo a Passo

```
1. Inspeção da Arquitetura:
   Substituir recursivamente todos os nn.ReLU da CNN por GuidedReLU_Module.

2. Forward Pass:
   A imagem passa pela rede. A cada ReLU modificada, o tensor de entrada x
   é armazenado na memória (ctx.save_for_backward).

3. Backward Pass com Filtro Duplo:
   O gradiente flui de trás para frente. Em cada GuidedReLU:
   - Zera se x <= 0 (neurônio estava inativo na ida).
   - Zera se grad <= 0 (gradiente estava diminuindo a predição na volta).

4. Extração no Nível de Imagem:
   gradiente_final = image.grad
   heatmap = max(|gradiente_final|, canais_RGB)
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`guided_backprop.py`](/guided_backprop.py):

```python
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x) # Salva a ativação da ida
        return x.clamp(min=0)   # ReLU normal

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Regra de Ouro do Guided Backprop:
        grad_input[x <= 0] = 0           # Inativo no forward
        grad_input[grad_output <= 0] = 0 # Gradiente negativo no backward

        return grad_input

def replace_relu(module):
    """Substitui recursivamente todo nn.ReLU pelo módulo GuidedReLU"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedReLU_Module())
        else:
            replace_relu(child)
```

### Detalhe Crucial de Engenharia:
A função `replace_relu` faz um transplante cirúrgico em tempo de execução nos nós da árvore de módulos do PyTorch, permitindo que qualquer modelo (como ResNet50) passe a executar a retropropagação guiada sem necessidade de retreinar pesos.

---

## 5. Vantagens e Limitações

### Vantagens:
- **Resolução e Nitidez Excepcionais:** Produz visualizações extremamente limpas, destacando bordas vasculares, costelas e silhuetas de órgãos sem o borrão de outros métodos.
- **Eficiência:** O custo computacional é o mesmo de um único passe backward.
- **Excelente quando combinado com SmoothGrad:** O `SmoothGrad(GuidedBackprop)` gera visualizações de altíssima fidelidade estética.

### Limitações e Controvérsias (The Sanity Checks Warning):
- **Independência parcial da classe (Adebayo et al., NeurIPS 2018):** Estudos fundamentais demonstraram que o Guided Backprop age em parte como um detector de bordas universal da imagem ("edge detector"), por vezes gerando mapas visuais semelhantes independentemente de qual classe seja retropropagada.
- **Portanto:** É excelente para entender **quais formas geométricas a rede detectou**, mas deve ser interpretado em conjunto com métodos CAM (como Grad-CAM) para confirmar a classe semântica.

---

## 6. Como Usar

```python
import torch
from guided_backprop import GuidedBackprop
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo pré-treinado carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

# Instanciação (substitui os ReLUs automaticamente)
explainer = GuidedBackprop(model, device)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Guided Backpropagation", show=True)
```
