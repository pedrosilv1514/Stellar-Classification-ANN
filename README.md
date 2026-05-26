# Período 2 – Redes Neurais Artificiais

> **Programa de Trainee CIS 2026 — IEEE Computational Intelligence Society (CIS) UnB**  
> Membro: Pedro Henrique Silva de Sousa

---

## Sobre o Projeto

Este repositório contém a solução completa da **Atividade Obrigatória do Período 2**, cujo objetivo é construir, treinar, otimizar e analisar criticamente uma Rede Neural Artificial utilizando **PyTorch** para a **classificação de corpos celestes** (Galáxias, Quasares e Estrelas) com base no [*Stellar Classification Dataset - SDSS17*](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

---

## Conteúdos Desenvolvidos

| Item | Descrição |
|------|-----------|
| **a** | Criação de uma rede neural de classificação utilizando PyTorch |
| **b** | Teste da variação de largura (neurônios) e profundidade (camadas) |
| **c** | Treinamento com diferentes épocas e *learning rates*, identificando *overfitting* e *underfitting* |
| **d** | Aplicação de Dropout, Weight Decay e otimizador Adam |

---

## Estrutura do Repositório

```
.
├── notebooks/
│   └── redes-neurais.ipynb 
├── data/
│   └── .gitkeep      
├── src/
│   └── model.py          
├── results/
│   └── .gitkeep          
├── requirements.txt
└── README.md
```

---

## Dataset

O dataset utilizado é o **Stellar Classification Dataset - SDSS17**, disponível publicamente no Kaggle:

```
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
```

**Para rodar localmente:** Faça o download do arquivo `star_classification.csv` e coloque-o dentro da pasta `data/`. O notebook detecta automaticamente o caminho e, caso o arquivo não seja encontrado, gera um **dataset sintético estruturalmente equivalente** para simulação.

> O arquivo `star_classification.csv` não está versionado neste repositório (está listado no `.gitignore`) por ser um arquivo de terceiros.

---

## Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Opção 1 — Jupyter local

```bash
jupyter notebook notebooks/redes-neurais.ipynb
```

### Opção 2 — Kaggle

O notebook está estruturado para rodar diretamente no Kaggle com o dataset vinculado. O caminho `/kaggle/input/datasets/fedesoriano/stellar-classification-dataset-sdss17/star_classification.csv` é detectado automaticamente.

---

## Resultados Esperados

Ao executar o notebook completo, serão gerados:

- 📊 Gráfico comparativo de loss por arquitetura (rasa, média, profunda)
- 📈 Curvas de aprendizado dos cenários de underfitting e overfitting
- ✅ Curvas de aprendizado do modelo final regularizado
- 🔵 Matriz de Confusão (Medição Física Real vs. Predição)
- 📉 Curva ROC Multiclasse por corpo celeste (GALAXY, QSO, STAR)
- 📋 Relatório técnico estatístico (`classification_report`)

---

## Arquitetura do Modelo Final

```
Input (10 features)
    └── Linear(10 → 128) + ReLU + Dropout(0.25)
    └── Linear(128 → 64)  + ReLU + Dropout(0.25)
    └── Linear(64 → 3)    → Logits
```

**Hiperparâmetros:** `epochs=50`, `lr=0.002`, `weight_decay=1e-4`, `optimizer=Adam`

---

## Dependências

| Pacote | Versão mínima |
|--------|--------------|
| torch | 2.0.0 |
| numpy | 1.24.0 |
| pandas | 2.0.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scikit-learn | 1.3.0 |

---

## Licença

Projeto acadêmico desenvolvido para o Programa de Trainee CIS 2026 — IEEE CIS UnB.