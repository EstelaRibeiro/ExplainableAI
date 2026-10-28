# Framework de Interpretabilidade e XAI para CNNs Médicas

Este repositório é uma suíte completa de **Inteligência Artificial Explicável (Explainable AI - XAI)** para Redes Neurais Convolucionais (CNNs) em visão computacional médica. O projeto implementa, otimiza e compara visualmente **11 abordagens de interpretabilidade**, permitindo entender de forma detalhada as decisões do modelo no diagnóstico de **Aorta Alongada** (e outras condições radiológicas).

---

## 📚 Índice da Documentação Didática (`DOCS/`)

Para uma explicação teórica aprofundada, fórmulas matemáticas explicadas passo a passo, intuições visuais e análise linha a linha do código de cada técnica, consulte os documentos dedicados na pasta [`DOCS/`](/DOCS/):

1. [**Visão Geral e Taxonomia de XAI**](/DOCS/00_visao_geral_e_taxonomia.md): Entenda as 3 grandes famílias (Gradiente, CAM e Perturbação).
2. [**Vanilla Saliency Maps**](/DOCS/01_saliency_maps.md): Sensibilidade linear direta de primeira ordem no nível de pixel.
3. [**SmoothGrad**](/DOCS/02_smoothgrad.md): Remoção estocástica de ruído via perturbações gaussianas.
4. [**Guided Backpropagation**](/DOCS/03_guided_backprop.md): Filtro duplo de ReLUs para visualização nítida de bordas anatômicas.
5. [**Integrated Gradients**](/DOCS/04_integrated_gradients.md): Atribuição axiomática via integração sobre linha de base neutra.
6. [**Grad-CAM**](/DOCS/05_gradcam.md): Ponderação semântica da última camada convolucional via Global Average Pooling.
7. [**Grad-CAM++**](/DOCS/06_gradcampp.md): Generalização com derivadas de 2ª e 3ª ordem para objetos múltiplos e irregulares.
8. [**Layer-CAM (LayerGradCAM)**](/DOCS/07_layercam.md): Ponderação espacial elemento a elemento para maior foco anatômico.
9. [**Score-CAM e Score-CAM Otimizado**](/DOCS/08_scorecam.md): Técnica livre de gradientes com aceleração por inferência em lotes (GPU).
10. [**Occlusion Sensitivity**](/DOCS/09_occlusion_sensitivity.md): Teste de perturbação causal com janela deslizante.
11. [**RISE**](/DOCS/10_rise.md): Amostragem aleatória de Monte Carlo com milhares de máscaras suaves.
12. [**LIME**](/DOCS/11_lime.md): Modelo substituto local (Ridge) sobre superpixels anatômicos gerados via SLIC.
13. [**Comparativo Completo e Guia de Decisão**](/DOCS/12_comparativo_e_guia_de_escolha.md): Matriz comparativa e árvore de decisão prática.

---

## 🗂️ Estrutura do Repositório

```text
estudando_xai/
├── DOCS/                        # Documentação didática completa (.md)
│   ├── 00_visao_geral_e_taxonomia.md
│   ├── 01_saliency_maps.md
│   ├── 02_smoothgrad.md
│   ├── 03_guided_backprop.md
│   ├── 04_integrated_gradients.md
│   ├── 05_gradcam.md
│   ├── 06_gradcampp.md
│   ├── 07_layercam.md
│   ├── 08_scorecam.md
│   ├── 09_occlusion_sensitivity.md
│   ├── 10_rise.md
│   ├── 11_lime.md
│   └── 12_comparativo_e_guia_de_escolha.md
│
├── config_xai.py                # Configurações globais (caminhos, folds, parâmetros)
├── models_class.py              # Fábrica de modelos via biblioteca timm
├── utils_xai.py                 # Utilitários de pré-processamento, sobreposição e visualização
├── main_xai_run_all.py          # Script principal de execução e comparação entre folds
│
├── # Scripts dos Métodos de XAI
├── saliency.py                  # Vanilla Saliency Maps
├── smoothgrad.py                # Envoltório SmoothGrad (compatível com Saliency e Guided BP)
├── guided_backprop.py           # Guided Backpropagation
├── ig.py                        # Integrated Gradients
├── gradcam.py                   # Grad-CAM
├── gradcampp.py                 # Grad-CAM++
├── layergradcam.py              # Layer-CAM (LayerGradCAM)
├── scorecam.py                  # Score-CAM básico
├── scorecam_otimizado.py        # Score-CAM vetorizado e otimizado para GPU
├── occlusion.py                 # Occlusion Sensitivity
├── rise.py                      # RISE (Randomized Input Sampling for Explanation)
├── lime.py                      # LIME com segmentação SLIC
│
├── requirements_xai.txt         # Dependências do projeto
├── aorta_alongada_images/       # Diretório local de imagens médicas para teste
├── models_aorta/                # Modelos treinados (.pth) por fold
└── results/                     # Saídas geradas (heatmaps e painéis comparativos)
```

---

## ⚙️ Pré-requisitos e Instalação

Recomenda-se utilizar um ambiente virtual Python (versão 3.11 OU 3.12) com suporte a GPU CUDA.

1. **Instale as dependências listadas no `requirements_xai.txt`:**
   ```bash
   pip install -r requirements_xai.txt
   ```

   *Principais bibliotecas utilizadas:*
   - `torch` e `torchvision`: Treinamento, autograd e tensores GPU.
   - `timm`: Modelos de visão de ponta (ResNet50, EfficientNet, ConvNeXt, etc.).
   - `scikit-image`: Algoritmo SLIC para superpixels no LIME.
   - `scikit-learn`: Regressão Ridge e cálculo de distância de cosseno.
   - `opencv-python` (`cv2`): Geração do mapa de calor colorido (`TURBO`), resize e blend ponderado.
   - `matplotlib`: Geração das figuras de comparação e gráficos.

---

## 🚀 Como Usar

### 1. Configuração dos Caminhos e Parâmetros (`config_xai.py`)

Antes de rodar a inferência, abra o arquivo [`config_xai.py`] e verifique se os caminhos dos modelos e imagens correspondem ao seu ambiente:

```python
# Diretório base
OUTPUT_DIR = r'C:\Users\estela.ribeiro\JupyterNotebooks\estudando_cnns'

# Modelos treinados por fold (Cross-Validation)
MODEL_NAMES = [
    "best_model_resnet50_fold_0_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_1_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_2_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_3_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_4_Aortic enlargement__crop_rees_all_1_q.pth",
]

MODEL_PATH = OUTPUT_DIR + r'\models_aorta\resnet50'
MODEL_STR = 'resnet50' # Arquitetura no timm
IMG_SIZE = (384, 384)
```

---

### 2. Execução Automatizada Completa (`main_xai_run_all.py`)

Para rodar **todos os métodos** em sequência para **todos os folds de validação cruzada**, execute no terminal:

```bash
python main_xai_run_all.py
```

#### O que este script faz automaticamente?
1. Detecta o dispositivo disponível (`CUDA` ou `CPU`).
2. Carrega a imagem médica alvo definida em `img_name`.
3. Para cada fold ($0$ a $4$):
   - Instancia o modelo ResNet50 e carrega os pesos treinados.
   - Executa cada explicador:
     - `saliency`
     - `guided_backprop`
     - `smoothgrad_guided_backprop`
     - `integrated_gradients`
     - `gradcam`
     - `gradcampp`
     - `layergradcam`
     - `scorecam_otim`
     - `occlusion`
     - `lime`
   - Salva a visualização individual de 3 colunas (Imagem Original | Heatmap | Sobreposição) em `results/<nome_do_modelo>/<imagem>/<metodo>.png`.
4. Constrói um **Grid Comparativo Gigante** consolidando todos os métodos nas linhas e os folds nas colunas, salvo em alta resolução (300 DPI) e formato vetorial EPS:
   - `results/<nome_da_imagem>_comparison.png`
   - `results/<nome_da_imagem>_comparison.eps`

---

### 3. Exemplo de Execução Individual de um Método

Se você deseja testar apenas um método específico (por exemplo, o **Layer-CAM** ou o **Score-CAM Otimizado**) de forma isolada em um script ou notebook interativo:

```python
import torch
import os
import config_xai
import utils_xai
from models_class import get_classification_model
from layergradcam import LayerGradCAM

# 1. Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Carregar modelo
model = get_classification_model("resnet50", pretrained=False, num_classes=1)
model_path = os.path.join(config_xai.MODEL_PATH, config_xai.MODEL_NAMES[0])
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# 3. Carregar e pré-processar imagem
img_dir = r"C:\Users\estela.ribeiro\JupyterNotebooks\estudando_cnns\aorta_alongada_images"
img_name = "0e8b19dc3af06aaed4718cd664a896bf_crop_rees_all_1_q.png"

sample = utils_xai.get_image(img_dir, img_name, config_xai.IMG_SIZE)
image_normalized = sample["normalized"]
image_path = os.path.join(img_dir, img_name)

# 4. Instanciar e gerar explicação
explainer = LayerGradCAM(model, device)
result = explainer.generate(image_normalized)

heatmap = result["heatmap"]
prob = result["probability"].item()
print(f"Probabilidade predita de Aorta Alongada: {prob:.4f}")

# 5. Criar sobreposição visual (Overlay) e exibir
overlay = utils_xai.overlay_heatmap(
    image_path,
    heatmap,
    alpha=0.65,
    gamma=0.6,
    threshold=0.35
)

fig = utils_xai.show_explanation(
    image_path,
    heatmap,
    overlay,
    method_name="Layer-CAM",
    show=True
)
```

---

## 🔬 Como Interpretar as Saídas Geradas

As imagens geradas em `results/` utilizam o colormap de alto contraste `TURBO` do OpenCV:

- 🔴 **Cores Quentes (Vermelho e Laranja):** Indicam as regiões de máxima importância atribuída pela rede. Para casos positivos de Aorta Alongada, espera-se que essas áreas concentrem-se ao redor da borda lateral do arco aórtico e do botão aórtico.
- 🟡 **Cores Médias (Amarelo e Verde):** Indicam regiões de relevância secundária ou contexto anatômico de suporte (silhueta cardíaca, mediastino).
- 🔵 **Cores Frias (Azul e Roxo):** Áreas neutras ou com contribuição nula para o diagnóstico.
- 🏷️ **Probabilidade:** No canto superior esquerdo de cada sub-painel de comparação, a probabilidade $P \in [0, 1]$ prevista pelo modelo naquele fold é exibida para correlacionar o mapa à certeza da predição.

---

## 🛠️ Boas Práticas e Recomendações Clínicas

1. **Nunca confie em um único método isoladamente:** Métodos baseados em gradientes finos (como Guided Backprop) são ótimos para verificar bordas, mas podem agir como filtros de borda genéricos. Combine-os sempre com um método da família CAM (Grad-CAM++, Layer-CAM ou Score-CAM) para garantir coerência semântica.
2. **Cuidado com Vieses Espúrios:** Se o mapa de calor destacar textos radiográficos (ex.: marcação "D" de direita), pontas de cateteres ou as bordas pretas do colimador do raio-X, o modelo pode ter aprendido atalhos não generalizáveis.
3. **Desempenho em GPU:** Métodos baseados em perturbação como `RISE` e `LIME` exigem centenas ou milhares de inferências. Certifique-se de executá-los em ambiente com aceleração CUDA habilitada.
