# Comparativo Completo e Guia de Escolha Prática

Este documento consolida a análise comparativa entre todos os métodos de **Inteligência Artificial Explicável (XAI)** implementados no projeto, fornecendo uma matriz de decisão para aplicações em visão computacional médica.

---

## 1. Matriz Comparativa Multidimensional

| Método | Família | Dependência de Gradiente | Resolução Espacial | Custo Computacional | Robustez ao Ruído | Fidelidade Causal |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| [**Saliency**](/DOCS/01_saliency_maps.md) | Gradiente puro | Sim (1ª ordem) | Pixel fino | Muito Baixo ($1$ backward) | Baixa (muito ruído) | Média |
| [**SmoothGrad**](/DOCS/02_smoothgrad.md) | Envoltório estocástico | Sim | Pixel fino filtrado | Médio ($N=30$ iterações) | Alta | Média-Alta |
| [**Guided Backprop**](/DOCS/03_guided_backprop.md) | Gradiente modificado | Sim | Bordas ultra-nítidas | Muito Baixo ($1$ backward) | Muito Alta | Baixa (age como filtro de bordas) |
| [**Integrated Gradients**](/DOCS/04_integrated_gradients.md) | Gradiente axiomático | Sim (integral) | Pixel fino | Médio ($m=50$ passos) | Média | Alta (satisfaz completude) |
| [**Grad-CAM**](/DOCS/05_gradcam.md) | Baseado em CAM | Sim (GAP) | Semântica grosseira | Muito Baixo ($1$ fwd + $1$ bwd) | Muito Alta | Alta |
| [**Grad-CAM++**](/DOCS/06_gradcampp.md) | Baseado em CAM | Sim (ordem 2 e 3) | Semântica detalhada | Muito Baixo ($1$ fwd + $1$ bwd) | Muito Alta | Alta |
| [**Layer-CAM**](/DOCS/07_layercam.md) | Baseado em CAM | Sim (ponto a ponto) | Semântica + Fina | Muito Baixo ($1$ fwd + $1$ bwd) | Muito Alta | Muito Alta |
| [**Score-CAM Otim.**](/DOCS/08_scorecam.md) | CAM livre de gradiente | Não (Forward score) | Semântica coesa | Médio (lotes de $64$) | Imune a gradiente | Muito Alta |
| [**Occlusion**](/DOCS/09_occlusion_sensitivity.md) | Perturbação mecânica | Não (Caixa-preta) | Blocos de janela | Alto ($~576$ inferências) | Alta | Altíssima (intervenção direta) |
| [**RISE**](/DOCS/10_rise.md) | Amostragem aleatória | Não (Caixa-preta) | Regiões orgânicas | Muito Alto ($4000$ inferências) | Alta | Altíssima (Monte Carlo) |
| [**LIME**](/DOCS/11_lime.md) | Substituto local | Não (Caixa-preta) | Superpixels anatômicos | Alto ($1000$ inferências) | Alta | Alta |

---

## 2. Árvore de Decisão: Qual Método Escolher?

Para definir qual técnica empregar em cada estágio do seu projeto ou análise clínica, siga o fluxograma lógico:

```
                            ¿O QUE VOCÊ PRECISA EXPLICAR?
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │                                                           │
   [ Contexto Clínico / Órgãos ]                               [ Inspeção de Bordas / Texturas ]
            │                                                           │
  ¿Qual o limite computacional?                               ¿Há ruído excessivo no gradiente?
      │                       │                                     │                       │
      ▼                       ▼                                     ▼                       ▼
  [ Tempo Real / Rápido ]   [ Alta Precisão Causal ]            [ Sim ]                  [ Não ]
      │                       │                                     │                       │
      ├─ Layer-CAM            ├─ Score-CAM Otimizado                └─ SmoothGrad +         ├─ Integrated Gradients
      ├─ Grad-CAM++           ├─ LIME (superpixels anatômicos)         Guided Backprop      └─ Vanilla Saliency
      └─ Grad-CAM             └─ RISE (se tempo permitir)
```

---

## 3. Recomendações Práticas para Imagens Médicas (Aorta Alongada)

1. **Para Comunicação com Médicos e Radiologistas:**
   - **Recomendado:** **Layer-CAM**, **Grad-CAM++** e **LIME**.
   - *Por quê?* Médicos raciocinam por estruturas anatômicas completas. O Layer-CAM e o Grad-CAM++ delimitam muito bem a extensão do arco da aorta sem vazar para o pulmão adjacente. O LIME respeita as fronteiras dos tecidos graças à segmentação SLIC.

2. **Para Validação Científica e Auditoria de Modelos (Passar nos "Sanity Checks"):**
   - **Recomendado:** **Score-CAM Otimizado** e **Integrated Gradients**.
   - *Por quê?* São métodos imunes a artefatos espúrios de gradiente ou matematicamente fundamentados em teoria dos jogos. Se o Score-CAM destacar a aorta, você tem a certeza matemática de que a ativação daquele canal de fato aumentou a probabilidade do modelo.

3. **Para Triagem Rápida em Datasets Grandes:**
   - **Recomendado:** **Grad-CAM** ou **Layer-CAM**.
   - *Por quê?* Ambos rodam a mais de 30 frames por segundo em GPU, permitindo gerar explicações para milhares de pacientes em minutos.

---

## 4. Como Interpretar Discrepâncias Entre Métodos?

É comum que diferentes métodos produzam heatmaps ligeiramente divergentes para a mesma imagem. Eis como interpretar essas diferenças:

- **Se o Grad-CAM destaca a aorta, mas o Vanilla Saliency parece ruído puro:**
  - *Diagnóstico:* É o comportamento esperado. O modelo aprendeu o conceito anatômico correto nas camadas convolucionais profundas, mas os gradientes locais de primeira ordem nos pixels estão ruidosos/saturados.
- **Se o LIME e a Oclusão destacam a mesma região que o Grad-CAM:**
  - *Diagnóstico:* **Concordância inter-método alta.** É o cenário de maior confiança clínica, pois tanto a inspeção interna de ativações (CAM) quanto a perturbação externa da imagem (Black-box) convergem para a mesma conclusão diagnóstica.
- **Se um método baseado em perturbação (ex.: Oclusão) destaca uma anomalia em um canto que não tem anatomia relevante (ex.: borda da imagem):**
  - *Diagnóstico:* **Atenção para atalhos espúrios (Clever Hans effect).** A rede pode estar utilizando a borda da chapa radiográfica ou marcadores de texto como pista para a predição.
