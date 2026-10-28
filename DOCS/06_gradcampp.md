# Grad-CAM++: Generalização com Derivadas de Ordem Superior

**Arquivo no projeto:** [`gradcampp.py`](/gradcampp.py)  
**Artigo pioneiro:** *Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks* (Chattopadhay, Sarkar, Howlader & Balasubramanian, IEEE WACV 2018 - arXiv:1710.11063)

---

## 1. Intuição Didática

O Grad-CAM padrão é excelente, mas tem um "calcanhar de Aquiles" bem conhecido: quando uma imagem contém **múltiplas ocorrências do mesmo objeto** ou quando a anomalia possui **áreas de tamanhos muito diferentes**, o cálculo de média simples (GAP) do Grad-CAM acaba "engolindo" as regiões menores e dando ênfase apenas ao ponto mais dominante.

Pense em uma radiografia com duas alterações pulmonares bilaterais: o Grad-CAM tradicional costuma destacar intensamente apenas a maior delas, ignorando a segunda.

O **Grad-CAM++** corrige essa deficiência:
> *"Em vez de calcular uma média aritmética simples e cega de todos os gradientes, vamos pesar a contribuição de cada pixel do mapa de ativação individualmente, usando derivadas de 2ª e 3ª ordem para capturar a curvatura e a não-linearidade local da rede."*

Isso permite que o Grad-CAM++ capture com muito mais fidelidade toda a extensão de estruturas anatômicas complexas (como o arco aórtico alongado em sua totalidade) e múltiplos focos patológicos.

---

## 2. Fundamentação Matemática

No Grad-CAM tradicional, o peso $\alpha_k^c$ é constante para todo o canal $k$. No **Grad-CAM++**, o peso final do canal $w_k^c$ é calculado como uma soma ponderada dos gradientes positivos:

$$w_k^c = \sum_{i=1}^{H'} \sum_{j=1}^{W'} \alpha_{i, j}^{kc} \cdot \text{ReLU}\left( \frac{\partial Y^c}{\partial A_{i, j}^k} \right)$$

Onde os coeficientes $\alpha_{i, j}^{kc}$ equilibram a importância de cada posição espacial $(i, j)$ no canal $k$. Através de uma dedução analítica rigorosa com aproximação de Taylor de ordem superior, os autores provaram que:

$$\alpha_{i, j}^{kc} = \frac{\left( \frac{\partial^2 Y^c}{\partial (A_{i, j}^k)^2} \right)}{2 \left( \frac{\partial^2 Y^c}{\partial (A_{i, j}^k)^2} \right) + \sum_{a, b} A_{a, b}^k \left( \frac{\partial^3 Y^c}{\partial (A_{a, b}^k)^3} \right)}$$

Na aproximação prática para redes convolucionais com classificadores lineares ou sigmoides, as potências do gradiente de primeira ordem são utilizadas como estimadores eficientes dessas ordens superiores:
- $\text{grads}^2 = \left( \frac{\partial Y^c}{\partial A_{i, j}^k} \right)^2$
- $\text{grads}^3 = \left( \frac{\partial Y^c}{\partial A_{i, j}^k} \right)^3$

Após obter os pesos $w_k^c$, o mapa final segue a mesma agregação do Grad-CAM:

$$L_{\text{Grad-CAM++}}^c = \text{ReLU} \left( \sum_k w_k^c A^k \right)$$

---

## 3. Comparativo Grad-CAM vs. Grad-CAM++

```
                     GRAD-CAM                          GRAD-CAM++
               ─────────────────────             ────────────────────────
Ponderação:    Média simples (GAP)               Média ponderada por α_ij
               de todos os gradientes            de gradientes estritamente positivos

Derivadas:     Apenas 1ª ordem                   1ª, 2ª e 3ª ordem (curvatura)

Comportamento: Tende a iluminar um único         Ilumina o objeto completo, bordas e
               ponto central dominante           múltiplas ocorrências simultâneas
```

---

## 4. Análise da Implementação no Projeto

No arquivo [`gradcampp.py`](/gradcampp.py):

```python
class GradCAMpp:
    ...
    def generate(self, image):
        ...
        score = output.squeeze()
        score.backward()

        activations = self.activations
        gradients = self.gradients

        # 1. Cálculo das potências do gradiente
        grads2 = gradients.pow(2)
        grads3 = gradients.pow(3)

        # 2. Denominador da formulação de ordem superior
        denominator = (2 * grads2 + (activations * grads3).sum(dim=(2, 3), keepdim=True))
        
        # Proteção contra divisão por zero
        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator)
        )

        # 3. Coeficientes espaciais alpha
        alpha = grads2 / denominator

        # 4. Considera apenas gradientes com contribuição positiva (ReLU)
        positive_gradients = F.relu(gradients)

        # 5. Pesos finais por canal
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)

        # 6. Combinação linear e ReLU
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # 7. Interpolação bilinear e normalização
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        cam = normalize(cam.squeeze())
```

---

## 5. Vantagens e Limitações

### Vantagens:
- **Melhor Delimitação de Contornos:** Capta a silhueta da estrutura anatômica de forma mais completa do que o Grad-CAM original.
- **Suporte a Múltiplos Focos:** Muito superior na identificação de lesões multifocais ou anomalias dispersas.
- **Eficiência Computacional:** Consegue calcular as derivadas de ordem superior analiticamente no mesmo passe de retropropagação, mantendo o tempo de execução quase idêntico ao do Grad-CAM.

### Limitações:
- **Resolução Semântica Ainda Condicionada:** Assim como o Grad-CAM, a base espacial ainda provém da última camada convolucional ($H' \times W'$ reduzido), dependendo de interpolação bilinear contínua.

---

## 6. Como Usar

```python
import torch
from gradcampp import GradCAMpp
from utils_xai import get_image, overlay_heatmap, show_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ... # Modelo carregado

sample = get_image("caminho/imagens", "exemplo.png", (384, 384))
image = sample["normalized"]

explainer = GradCAMpp(model, device)
result = explainer.generate(image)

heatmap = result["heatmap"]

# Visualizar
overlay = overlay_heatmap("caminho/imagens/exemplo.png", heatmap)
fig = show_explanation("caminho/imagens/exemplo.png", heatmap, overlay, method_name="Grad-CAM++", show=True)
```
