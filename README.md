# NLP – Análise de Sentimentos em Tweets em Português 🇧🇷

Projeto completo de **Processamento de Linguagem Natural (PLN)** para **análise de sentimento** em português usando o dataset do Kaggle **“Portuguese Tweets for Sentiment Analysis”** (positivo/negativo/neutro). O projeto inclui todas as etapas da ciência de dados, com **pré-processamento**, **tokenização**, **normalização**, **remoção de símbolos**, **stopwords**, **stemming e lematização**, vetorização com **TF‑IDF** e **caracteres**, criação de **embeddings** (*FastText* via *gensim*), **seleção/treinamento de modelos**, **comparação por métricas**, explicabilidade com **LIME** e **deploy** via CLI e **API FastAPI**.

**Dataset:** Kaggle – *Portuguese Tweets for Sentiment Analysis*. Classes: *positive*, *negative*, *neutral*. Os arquivos incluem colunas como `id`, `tweet_text`, `tweet_date`, `sentiment`, `query_used` (conforme descrito por repositórios que utilizam esse dataset).

---

## 🔧 Stack e principais bibliotecas

- Python 3.9+
- `pandas`, `numpy`
- `scikit-learn` (modelos clássicos + métricas)
- `nltk` (stopwords e stemmer RSLP pt-br)
- `spacy` (opcional, lematização com `pt_core_news_sm`)
- `gensim` (embeddings FastText treinados no próprio corpus)
- `joblib` (persistência de artefatos)
- `matplotlib` (gráficos de avaliação)
- `kaggle` (download do dataset via API)
- `lime` (explicabilidade por instância)
- `fastapi` + `uvicorn` (serviço de inferência)

> Extras opcionais: `xgboost`, `torch`/`tensorflow` e `transformers` (para modelos mais pesados).

---

## 🗂️ Estrutura de Pastas e Arquivos

```
nlp_pt_sa/
├── README.md
├── requirements.txt
├── config/
│   └── config.yml.txt
├── data/
│   ├── raw/          # CSVs originais do Kaggle (após download)
│   └── processed/    # Dados limpos e prontos p/ modelagem
├── models/
│   ├── artifacts/    # Vetorizadores, modelos e pipeline salvos
│   └── reports/      # Métricas, gráficos e leaderboard
├── scripts/          # Cada etapa do pipeline em .txt (executável em Python)
│   ├── 00_setup_env.txt
│   ├── 01_download_kaggle.txt
│   ├── 02_preprocess.txt
│   ├── 03_vectorize_train_baselines.txt
│   ├── 04_embeddings_fasttext_train.txt
│   ├── 05_evaluate_compare.txt
│   ├── 06_inference_cli.txt
│   └── 07_api_fastapi.txt
└── utils/
    ├── text_utils.txt
    └── ml_utils.txt
```

> **Observação:** Todos os **códigos** do pipeline estão em **`.txt`** conforme solicitado. O Python consegue executar arquivos `.txt` normalmente: `python scripts/02_preprocess.txt`. Se preferir, renomeie para `.py`.

---

## ⚙️ Instalação rápida

1) **Crie e ative** um ambiente virtual
```bash
python -m venv .venv
# Windows: 
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) **Instale dependências**
```bash
pip install -r requirements.txt
```

3) **Baixe recursos NLTK e modelo spaCy (opcional, recomendado)**
```bash
python scripts/00_setup_env.txt --with-spacy
```

4) **Configure a API do Kaggle**  
Crie `~/.kaggle/kaggle.json` com suas credenciais ou defina `KAGGLE_USERNAME` e `KAGGLE_KEY` no ambiente.

5) **Baixe o dataset**
```bash
python scripts/01_download_kaggle.txt
```

6) **Pré-processamento (tokenização, normalização, stopwords, stemming/lematização)**  
```bash
python scripts/02_preprocess.txt --strip-accents false --lemmatize true --stem false
```

7) **Treinar modelos de baseline (TF‑IDF / char n‑grams)**
```bash
python scripts/03_vectorize_train_baselines.txt --models mnb,logreg,svm
```

8) **Embeddings (FastText via gensim) + modelos**
```bash
python scripts/04_embeddings_fasttext_train.txt --models logreg,svm
```

9) **Consolidar resultados e escolher o melhor**
```bash
python scripts/05_evaluate_compare.txt
```

10) **Inferência por CLI**
```bash
python scripts/06_inference_cli.txt --text "Gostei muito do atendimento, excelente!"
```

11) **API FastAPI**
```bash
uvicorn scripts.07_api_fastapi:app --reload
# ou
python scripts/07_api_fastapi.txt
```

---

## 🧪 Modelos testados e métricas

- **Multinomial Naive Bayes** (bag-of-words / TF‑IDF)
- **Logistic Regression** (TF‑IDF e embeddings)
- **Linear SVM** (TF‑IDF e embeddings)

Métricas reportadas (validação e teste): *accuracy*, *precision/recall/F1 macro e weighted*, *AUC micro/macro (multiclasse, quando aplicável)*, *matriz de confusão* e *leaderboard* consolidado.

---

## 🧠 Pré-processamento e PLN (resumo do que os scripts fazem)

- **Tokenização:** spaCy (pt) se disponível; fallback com regex/NLTK.
- **Normalização:** lowercasing, URLs/menções/hashtags, risadas (“kkkk”), repetição de letras, números e pontuação.
- **Stopwords:** NLTK + lista customizada (“rt”, “via”, etc.).
- **Stemming/Lematização:** RSLPStemmer (NLTK) e/ou lematização spaCy (`pt_core_news_sm`).
- **Vetorização:** TF‑IDF (palavras e caracteres), *n‑grams*, `min_df`/`max_df` configuráveis.
- **Embeddings:** FastText (gensim) treinado no corpus → vetor de documento por média de vetores de palavras.
- **Seleção de modelo:** *GridSearchCV* com validação estratificada, `class_weight='balanced'` quando suportado.
- **Persistência:** modelos/vecs em `models/artifacts/` + métricas em `models/reports/`.

---

## 🔎 Sobre o dataset

- Repositório Kaggle: *Portuguese Tweets for Sentiment Analysis* – tweets em **português** rotulados em **positivo**, **negativo** e **neutro**.
- Colunas (exemplos comuns no conjunto): `id`, `tweet_text`, `tweet_date`, `sentiment`, `query_used`.
- O pipeline detecta automaticamente o arquivo principal em `data/raw/` e a coluna de texto/label, com *fallbacks* configuráveis em `config/config.yml.txt`.

---

## ✍️ Licença

MIT – use livremente com atribuição. Dados do Kaggle seguem a licença do fornecedor.
