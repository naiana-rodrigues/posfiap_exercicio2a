# Predição de Câncer de Mama com Machine Learning e LLM

## Objetivo

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de classificar casos de câncer de mama como benignos ou malignos utilizando o dataset Breast Cancer Wisconsin.

Além da construção dos modelos preditivos, foi realizada a integração com uma LLM para interpretação dos resultados, análise dos modelos e geração de explicações em linguagem natural.

---

# Base de Dados

Dataset utilizado:

**Breast Cancer Wisconsin Dataset**

Fonte:

Kaggle - `uciml/breast-cancer-wisconsin-data`

Características:

- 569 amostras
- 30 atributos originais relacionados às características dos tumores
- Variável objetivo:
  - M (Maligno) → 1
  - B (Benigno) → 0

---

# Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- KaggleHub
- Groq API (Llama)
- Google Gemini API

---

# Etapas do Projeto

## 1. Preparação dos Dados

Foram realizados:

- Importação da base utilizando KaggleHub;
- Remoção da coluna sem dados (`Unnamed: 32`);
- Conversão do diagnóstico para variável numérica;
- Separação dos dados em treino (80%) e teste (20%);
- Uso de estratificação para preservar a proporção das classes.

---

# Questões Norteadoras da Análise

As análises foram conduzidas a partir das seguintes perguntas:

## Como o tamanho da base influencia a confiabilidade das previsões?

A quantidade e representatividade dos dados influenciam a capacidade de generalização do modelo.

Apesar dos bons resultados obtidos, uma aplicação real exigiria bases maiores, mais diversificadas e validadas clinicamente.

---

## As características dos tumores apresentam padrões capazes de diferenciar casos benignos e malignos?

A análise de correlação demonstrou relações importantes entre as variáveis, indicando padrões que podem auxiliar os algoritmos na classificação.

---

## Por que comparar diferentes modelos de Machine Learning?

Cada algoritmo aprende os padrões dos dados de maneira diferente.

A comparação permite selecionar o modelo mais adequado considerando:

- desempenho estatístico;
- capacidade de generalização;
- impacto dos erros.

---

# Análise Exploratória

Foram realizadas:

- Estatísticas descritivas;
- Matriz de correlação;
- Heatmap geral;
- Heatmap considerando somente casos malignos.

Essas análises permitiram identificar relações entre as características dos tumores e apoiar a etapa de modelagem.

---

# Modelos Desenvolvidos

## 1. Regressão Logística

Modelo utilizado para classificação binária entre casos benignos e malignos.

Processamento aplicado:

- Remoção de variáveis com baixa influência;
- Remoção de identificadores;
- Remoção da variável original de diagnóstico;
- Padronização utilizando `StandardScaler`.

Configuração:

```python
LogisticRegression(max_iter=5000)
```

Resultado:

```
Accuracy: 0.9737

Precision: 0.9756

Recall: 0.9524

F1-score: 0.9639
```

Matriz de Confusão:

```
[[71,1],
 [2,40]]
```

O modelo apresentou apenas 2 falsos negativos.

---

## 2. Árvore de Decisão

Modelo desenvolvido para comparação devido à sua capacidade de interpretação.

Configuração inicial:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=2
)
```

Após análise utilizando LLM, foram realizados testes de hiperparâmetros.

Melhor configuração encontrada:

```python
DecisionTreeClassifier(
    max_depth=8
)
```

Resultado:

```
Accuracy: 0.9298

ROC-AUC: 0.9228

F1-score: 0.9000
```

Matriz de Confusão:

```
[[70,2],
 [6,36]]
```

---

# Comparação dos Modelos

| Modelo | Accuracy | F1-score | Falsos Negativos |
|---|---|---|---|
| Regressão Logística | 0.9737 | 0.9639 | 2 |
| Árvore de Decisão | 0.9298 | 0.9000 | 6 |

## Conclusão da Comparação

A Regressão Logística apresentou melhor desempenho geral para o conjunto de dados utilizado.

O modelo apresentou:

- maior Accuracy;
- maior Precision;
- maior Recall;
- maior F1-score;
- menor quantidade de falsos negativos.

Por esse motivo, foi escolhido como modelo final.

---

# Importância das Variáveis

Foi realizada análise das características que mais influenciaram os modelos.

Principais atributos identificados:

- `perimeter_worst`
- `concave points_worst`
- `texture_mean`
- `texture_worst`
- `area_worst`
- `radius_worst`

Essa análise contribui para aumentar a interpretabilidade dos resultados.

---

# Integração com LLM

Foram utilizadas LLMs pré-treinadas:

- Llama 3.3 via Groq;
- Gemini.

A integração teve como objetivo:

- interpretar resultados dos modelos;
- analisar métricas;
- sugerir melhorias;
- avaliar hiperparâmetros;
- gerar explicações em linguagem natural.

---

# Prompt Engineering

Foram criados prompts estruturados contendo:

- contexto do problema;
- características da base;
- resultados das métricas;
- matriz de confusão;
- importância das variáveis;
- objetivo específico da análise.

A estrutura dos prompts buscou direcionar a LLM para respostas técnicas e contextualizadas.

---

# Avaliação da LLM

A utilização da LLM contribuiu em três etapas:

## 1. Interpretação dos resultados

A LLM auxiliou na análise das métricas e identificação de pontos fortes e limitações dos modelos.

---

## 2. Sugestão de melhorias

Foram avaliadas sugestões de ajustes nos hiperparâmetros da Árvore de Decisão.

As configurações sugeridas foram testadas e comparadas com o modelo inicial.

---

## 3. Explicação dos resultados para usuários

Foi desenvolvido um fluxo onde o usuário informa as características do tumor e recebe:

- classificação prevista;
- probabilidade estimada;
- explicação em linguagem natural.

A resposta gerada é considerada apoio à decisão e não substitui avaliação médica.

---

# Arquitetura da Solução

Fluxo desenvolvido:

```
Usuário
   |
   | Entrada das características do tumor
   |
Modelo Machine Learning
   |
   | Classificação + Probabilidade
   |
LLM
   |
   | Explicação em linguagem natural
   |
Usuário
```

---

# Limitações

- Dataset pequeno para aplicação comercial;
- Necessidade de validação clínica;
- Modelo não substitui avaliação médica;
- Resultados dependem da qualidade dos dados utilizados.

---

# Melhorias Futuras

- Implementação de validação cruzada;
- Criação de API REST;
- Containerização com Docker;
- Implementação de Kubernetes para escalabilidade;
- Monitoramento e logging;
- Dashboard para acompanhamento;
- Utilização de bases maiores;
- Integração futura com dados textuais.

---

# Considerações sobre Escalabilidade

Como evolução da solução, a arquitetura poderá ser adaptada para ambiente de nuvem utilizando:

- containers Docker;
- Kubernetes para gerenciamento de cargas;
- mecanismos de escalabilidade automática;
- monitoramento de desempenho;
- registro de logs das predições realizadas.

A implementação em nuvem não foi realizada nesta etapa, sendo considerada uma melhoria futura.

---

# Conclusão

O projeto demonstrou que modelos tradicionais de Machine Learning podem apresentar bons resultados na classificação de câncer de mama.

A Regressão Logística apresentou o melhor desempenho para este conjunto de dados e foi escolhida como modelo final.

A integração com LLM agregou valor ao projeto ao permitir:

- interpretação dos resultados;
- análise crítica dos modelos;
- apoio na otimização;
- geração de explicações em linguagem natural.

Essa abordagem prepara a solução para futuras evoluções envolvendo dados textuais e aplicações mais completas na área da saúde.

---

# Naiana Rodrigues Pereira

Projeto acadêmico de Machine Learning aplicado à área da saúde.