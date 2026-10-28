# Visão Geral e Taxonomia de Inteligência Artificial Explicável (XAI)

Bem-vindo ao guia de métodos de **Inteligência Artificial Explicável (Explainable AI - XAI)** deste projeto! Este documento serve como porta de entrada conceitual para compreender como diferentes técnicas desvendam as decisões tomadas por Redes Neurais Convolucionais (CNNs) em tarefas de visão computacional e diagnóstico por imagem médica (ex.: detecção de Aorta Alongada).

---

## 1. O que é XAI e por que é essencial?

Modelos modernos de Deep Learning, como ResNet50 e EfficientNet, são comumente tratados como **"caixas-pretas" (black-boxes)**: eles recebem uma imagem de entrada de alta dimensão ($384 \times 384 \times 3$) e geram uma probabilidade ou classe predita, após milhões de operações matemáticas não lineares.

No contexto médico ou de tomada de decisão crítica, apenas saber **o que** a rede previu não é suficiente. Precisamos responder:
- **Onde** na imagem a rede prestou atenção?
- O modelo identificou biomarcadores anatômicos reais (ex.: dilatação/alongamento do arco aórtico) ou aprendeu artefatos espúrios (ex.: marcações do aparelho de raio-X, fios, bordas)?
- Podemos confiar na decisão para suporte ao diagnóstico clínico?

A área de **XAI** fornece ferramentas para mapear a importância de cada região da imagem para a decisão do modelo, gerando os chamados **Mapas de Calor de Atribuição (Heatmaps / Saliency Maps)**.

---

## 2. Taxonomia dos Métodos Implementados

Os métodos presentes neste repositório podem ser agrupados em três grandes famílias principais, dependendo de **como extraem a informação** do modelo:

```
                            ┌───────────────────────────────────────────────┐
                            │          MÉTODOS DE INTERPRETABILIDADE (XAI)   │
                            └───────────────────────┬───────────────────────┘
                                                    │
         ┌──────────────────────────────────────────┼────────────────────────────────────────┐
         │                                          │                                        │
┌────────▼──────────────┐             ┌─────────────▼──────────────┐             ┌───────────▼─────────────┐
│ Baseados em Gradiente │             │  Baseados em CAM / Ativação│             │ Baseados em Perturbação │
│   (Pixel-level)       │             │       (Feature-level)      │             │     (Model-Agnostic)    │
└────────┬──────────────┘             └─────────────┬──────────────┘             └───────────┬─────────────┘
         │                                          │                                        │
         ├─ Vanilla Saliency                        ├─ Grad-CAM                              ├─ Occlusion Sensitivity
         ├─ SmoothGrad                              ├─ Grad-CAM++                            ├─ RISE
         ├─ Guided Backpropagation                  ├─ Layer-CAM (LayerGradCAM)              └─ LIME
         └─ Integrated Gradients                    └─ Score-CAM / Score-CAM Otim.
```

### A. Métodos Baseados em Gradiente (Pixel-Level Attribution)
- **Princípio:** Calculam a derivada da pontuação da classe em relação aos pixels da imagem de entrada via backpropagation ($\frac{\partial Y_c}{\partial X}$).
- **Resolução:** Altíssima resolução (nível de pixel individual).
- **Características:** Destacam bordas e texturas locais, mas sofrem frequentemente com ruído visual e saturação de gradientes.
- **Métodos no projeto:**
  - [Vanilla Saliency Maps](/DOCS/01_saliency_maps.md)
  - [SmoothGrad](/DOCS/02_smoothgrad.md)
  - [Guided Backpropagation](/DOCS/03_guided_backprop.md)
  - [Integrated Gradients](/DOCS/04_integrated_gradients.md)

### B. Métodos Baseados em Ativação de Camadas (CAM Family)
- **Princípio:** Inspecionam os mapas de ativação da última camada convolucional ($A^k$), ponderando a importância de cada canal para a classe predita.
- **Resolução:** Resolução semântica grosseira (ex.: $12 \times 12$ ou $7 \times 7$), que é posteriormente interpolada via bilinear para o tamanho original da imagem.
- **Características:** Focam em regiões conceituais completas (órgãos, lesões, estruturas anatômicas), sendo muito estáveis e altamente legíveis clinicamente.
- **Métodos no projeto:**
  - [Grad-CAM](/DOCS/05_gradcam.md) (Pesos via média global dos gradientes)
  - [Grad-CAM++](/DOCS/06_gradcampp.md) (Pesos via derivadas de 2ª e 3ª ordem para múltiplos objetos)
  - [Layer-CAM / LayerGradCAM](/DOCS/07_layercam.md) (Ponderação espacial elemento a elemento)
  - [Score-CAM / Score-CAM Otimizado](/DOCS/08_scorecam.md) (Ponderação livre de gradientes por inferência direta)

### C. Métodos Baseados em Perturbação e Model-Agnostic
- **Princípio:** Não dependem da arquitetura interna ou de gradientes. Oclusões, máscaras aleatórias ou superpixels são aplicados à imagem, e a variação na probabilidade de saída do modelo é monitorada.
- **Resolução:** Determinada pelo tamanho da janela de oclusão, grade de máscaras ou superpixels.
- **Características:** Altamente fiéis ao comportamento real de caixa-preta, sem risco de artefatos de gradiente, mas computacionalmente mais exigentes (requerem dezenas a milhares de passes forward).
- **Métodos no projeto:**
  - [Occlusion Sensitivity](/DOCS/09_occlusion_sensitivity.md)
  - [RISE (Randomized Input Sampling for Explanation)](/DOCS/10_rise.md)
  - [LIME (Local Interpretable Model-agnostic Explanations)](/DOCS/11_lime.md)

---

## 3. Comparativo Rápido de Características

| Método | Família | Depende de Gradiente? | Granularidade | Custo Computacional |
| :--- | :--- | :---: | :--- | :---: |
| **Saliency** | Gradiente | Sim | Pixel (Fino) | Baixo (1 backward) |
| **SmoothGrad** | Gradiente | Sim | Pixel (Fino limpo) | Médio ($N$ forwards/backwards) |
| **Guided Backprop** | Gradiente | Sim | Pixel (Bordas finas) | Baixo (1 backward) |
| **Integrated Gradients** | Gradiente | Sim | Pixel (Fino fundamentado) | Médio ($M$ passos de interpolação) |
| **Grad-CAM** | CAM / Ativação | Sim | Regional / Semântica | Baixo (1 forward + 1 backward) |
| **Grad-CAM++** | CAM / Ativação | Sim | Regional detalhado | Baixo (1 forward + 1 backward) |
| **Layer-CAM** | CAM / Ativação | Sim | Regional + Fino | Baixo (1 forward + 1 backward) |
| **Score-CAM** | CAM / Perturbação | Não | Regional | Médio/Alto ($K$ canais convolucionais) |
| **Occlusion** | Perturbação | Não | Regional por blocos | Alto (depende do stride) |
| **RISE** | Perturbação | Não | Regional suave | Alto (milhares de máscaras) |
| **LIME** | Perturbação / Surrogate | Não | Superpixels anatômicos | Alto (centenas de amostragens) |

---

## 4. Como Navegar nesta Documentação

Cada arquivo nesta pasta `DOCS/` detalha um método específico com a seguinte estrutura padronizada:
1. **Intuição Didática**: O conceito em linguagem clara com analogias práticas.
2. **Fundamento Teórico e Formulação**: A matemática passo a passo e o porquê de cada operação.
3. **Fluxo do Algoritmo**: Diagrama e passos lógicos.
4. **Análise do Código Fonte no Projeto**: Exame linha a linha do script Python correspondente.
5. **Vantagens e Desvantagens**: Comparativo crítico e pontos de atenção.
6. **Exemplo Prático de Uso**: Código mínimo reproduzível.
7. **Referências Oficiais**: Paper original publicado.

Para prosseguir, escolha o método de interesse no índice acima ou siga a ordem numérica sugerida.
