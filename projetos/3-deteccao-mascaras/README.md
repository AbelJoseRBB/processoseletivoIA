# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 📝 Relatório do Candidato

👤 **Nome Completo: Abel José Rocha Barros Bezerra**

### 1️⃣ Resumo da Abordagem

Para a realização do fine-tuning foram os seguintes:

```python
    model.train(
        data = "dataset/data.yaml",
        epochs = 30,
        batch = 12,
        device = "cpu",
        imgsz = 640,
        workers = 4,

        mosaic = 1.0,
        fliplr = 0.5,

        name = "train",
        exist_ok = True
    ) 
``` 
Foi utilizado técnicas ligeiras de Data Augmentation com `mosaic = 1.0` e `fliplr = 0.5` para combater o desbalanceamento das classes, sem sobrecarregar o treinamento.

### 2️⃣ Bibliotecas Utilizadas

- **Ultralytics (v8.4.104):** Framework principal para treino e inferência.

- **PyTorch (torch v2.13.0):** Backend matemático do treino.

- **Torchvision:** Dependência nativa para operações visuais (NMS).

- **Google LiteRT (litert-torch>=0.9.0 e ai-edge-litert>=2.1.4):** Bibliotecas exigidas para a conversão de modelos Edge AI para o formato TFLite/LiteRT.

### 3️⃣ Técnica de Otimização do Modelo

A exportação foi realizada a partir da função:

```python
    model.export(format= "tflite", int8 = True, data = "dataset/data.yaml")
```

Utilizando a técnica de Quantização INT8 foi possível calibrar o modelo usando os dados de validação para converter os pesos de alta precisão para inteiros de 8 bits, assim, obtendo uma redução drástica de tamanho e aumento de velocidade.

### 4️⃣ Resultados Obtidos

O treinamento atingiu os seguintes resultados:

**Modelo original - Model.pt**

`mAP50 global`: 0.784

`mAP50-95 global`: 0.547

**Modelo otimizado - Model.tflite**

`mAP50 global`: 0.677

`mAP50-95 global`: 0.368

**Desempenho por Classe**

`with_mask`: 0.972

`without_mask`: 0.798

`mask_weared_incorrect`: 0.581

**Tamanhos**

`model.pt`: 5.3 MB

`model.tflite`: 2.9 MB

### 5️⃣ Comentários Adicionais (Opcional)

Neste projeto, pude perceber a importância de trabalhar em um ambiente padronizado e bem configurado, uma vez que parte do desenvolvimento envolveu lidar com conflitos de dependências, configuração do ambiente de execução e ajustes das bibliotecas necessárias para o treinamento e otimização do modelo.

Outro ponto importante a se notar foi o impacto negativo do desbalanceamento do conjunto de dados nos resultados obtidos, provocando por busca de técnicas para mitigar essa limitação, como a Data Augmentation, forçando o modelo a extrair mais padrões visuais da pouca quantidade de exemplos da classe `mask_weared_incorrect`.

### 6️⃣ Exemplo de Inferência

```
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss105.jpg                         11  [11x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss11.jpg                          43  [42x with_mask, 1x mask_weared_incorrect]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss113.jpg                         10  [9x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss12.jpg                          19  [16x with_mask, 3x without_mask]
----------------------------------------------------------------------
TOTAL                                       84
```
Com base nos dados obtidos, é possível observar uma superioridade da quantidade de exemplos da classe `with_mask` perante as outras classes, fato que acarreta em desbalanceamento nos resultados finais. 

Pelos exemplos analisados, em `inferencia_exemplos/predicoes`, os resultados foram obtidos tendo uma acurácia satistfatório, porém chamando a atenção para um "erro", na qual a análise foi realizada em um imagem na parede.  

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
