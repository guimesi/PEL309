# LID‑22 — Identificação de Idiomas em 22+ Línguas 🌍

Pipeline **reprodutível** de *Language Identification (LID)* cobrindo 22+ línguas, com foco em **simplicidade, robustez e rastreabilidade**. O projeto entrega um fluxo completo: **EDA**, *split* **estratificado por grupos** (hash do texto normalizado, para evitar vazamento), seleção de modelos via **GridSearchCV / HalvingGridSearchCV**, **diagnósticos anti-overfitting** (OOF, *learning curve*, **Y‑scramble**), **métricas por classe e por grupo de *script*** (Latin/CJK/Árabe etc.), **Top‑3 accuracy**, salvamento de **artefatos e metadados**.

> **Código principal:** `lid22.py` (executável direto)  
> **Dataset:** Kaggle — *language-identification-datasst* (Zara Jamshaid). O download é automático via **kagglehub** ou você pode definir o caminho manualmente via `LANGID_CSV`.

---

## 🔧 Stack (núcleo) e opcionais

- Python 3.9+  
- **Núcleo:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `kagglehub`
- **Opcionais (ativados automaticamente se instalados):**
  - `gensim` → **Word2Vec/Doc2Vec**
  - `xgboost` → **XGBoost** (com `XGB_USE_GPU=1` para `gpu_hist`, se houver GPU)

---

## 🗂️ Estrutura de saídas (artefatos)

Ao rodar o script, a pasta `artifacts/` é criada com:

```
artifacts/
├── run_YYYYmmdd-HHMMSS.txt                 # log completo da execução
├── plots_YYYYmmdd-HHMMSS/                  # todas as figuras geradas (EDA, matrizes, etc.)
├── cv_results_<experimento>.csv            # resultados do GridSearch por experimento
├── validation_results.csv                   # leaderboard de validação (macro‑F1)
├── classification_report_test_<best>.json   # relatório no TEST
├── errors_test_<best>.csv                   # amostras de erros (por par y_true/y_pred)
├── best_langid_<best>.joblib                # pipeline final salvo (train+val → test)
├── diag_text_leakage.json                   # duplicatas e *leakage* entre splits
├── oof_report_<best>.json                   # OOF (F1 macro + relatório)
├── learning_curve_<best>.csv                # learning curve (F1 macro) + figura
├── diag_y_scramble_<best>.csv               # distribuição de scores com rótulos embaralhados
└── metadata_ext.json                        # metadados consolidados do experimento
```

> As figuras (EDA, matrizes de confusão “bruta” e normalizada, confusão por *script*, barras de F1 etc.) entram em `plots_*` automaticamente.

---

## 🧠 Modelos e representações incluídos

**Sempre disponíveis**  
- **TF‑IDF (caracteres 2–5)** + **LinearSVC**  
- **TF‑IDF (palavras 1–2)** + **LogisticRegression**

**Se as libs estiverem instaladas**  
- **CountVectorizer (caracteres)** + **MultinomialNB**  
- **Word2Vec**/**Doc2Vec** (via `gensim`) + **LogisticRegression**  
- **TF‑IDF (palavras)** → **SVD** → **XGBoost** (multi‑classe), com suporte a **GPU**

Cada experimento é ajustado por *grid* e comparado por **macro‑F1** na validação. O **melhor** é *refit* em **train+val** e avaliado no **TEST**.

---

## 📊 Métricas e relatórios

- **Accuracy**, **F1 macro/micro** (val/test)  
- **Top‑3 accuracy**  
- **Relatório por classe** + **matriz de confusão** (bruta e normalizada)  
- **Métricas agregadas por *script*** (Latin, CJK, Árabe etc.) e **confusão por *script***  
- **Anti-overfitting:** OOF (*out‑of‑fold*), *learning curve*, **Y‑scramble** (múltiplas repetições)

---

## ⚙️ Instalação

```bash
# 1) Crie um ambiente
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Instale dependências
pip install -U pandas numpy scikit-learn matplotlib joblib kagglehub
# (opcionais)
pip install -U gensim xgboost
```

> Para baixar do Kaggle via `kagglehub`, configure suas credenciais do Kaggle (ou baixe o CSV manualmente e aponte `LANGID_CSV`).

---

## ▶️ Execução rápida

### (A) Usando download automático (kagglehub)
```bash
python lid22.py
```

### (B) Indicando o CSV localmente
```bash
# Windows (PowerShell)
$env:LANGID_CSV="C:\caminho\para\language.csv"; python lid22.py

# macOS/Linux
LANGID_CSV=/caminho/para/language.csv python lid22.py
```

Durante a execução você verá: amostras, colunas detectadas, EDA, *grid* por experimento, leaderboard de validação, avaliação final no **TEST**, diagnósticos anti‑overfitting e caminhos dos artefatos salvos.

---

## 🧩 Variáveis de ambiente (principais)

| Variável            | Default | Descrição |
|---|---:|---|
| `LANGID_CSV`        | —      | Caminho para o CSV (se **não** quiser usar `kagglehub`). |
| `RUN_HEAVY`         | `1`    | Ativa modelos “pesados”/extras (NB, W2V/D2V, XGB…). Use `0` p/ rodar só o essencial. |
| `SKIP_EDA`          | `0`    | Pula as análises/plots exploratórios. |
| `CV_FOLDS`          | `3`    | Número de dobras no CV. |
| `SEARCH_SUBSAMPLE`  | `0`    | Se >0, usa apenas N exemplos do *train* no *grid* (rápido p/ *smoke test*). |
| `GS_N_JOBS`         | `CPU`  | *Workers* no *grid*. (`loky`) |
| `USE_HALVING`       | `0`    | Usa `HalvingGridSearchCV` (se disponível). |
| `SPLIT_BY_GROUPS`   | `1`    | *Split* 80/10/10 estratificado por **grupos (hash do texto)** → evita vazamento. |
| `ANTI_OVERFITTING`  | `1`    | Executa OOF, *learning curve* e **Y‑scramble**. |
| `Y_SCRAMBLE_N`      | `5`    | Nº de repetições no Y‑scramble. |
| `CALIB_AT_END`      | `0`    | Calibra **LinearSVC** com `CalibratedClassifierCV` (sigmoid, cv=3). |
| `XGB_USE_GPU`       | `0`    | Define `gpu_hist`/`gpu_predictor` no XGBoost (se GPU disponível). |
| `W2V_WORKERS`       | `1`    | *Workers* para treinar Word2Vec/Doc2Vec. |
| `SKL_CACHE`         | —      | Cache do `Pipeline`. No Windows é **desabilitado** por padrão p/ evitar *PicklingError*. |

---

## 🧪 Exemplo de inferência (fora do `main`)

Depois de uma execução, carregue o melhor pipeline salvo:

```python
import joblib
pipe = joblib.load("artifacts/best_langid_<melhor_experimento>.joblib")

textos = [
    "A vida é bela e a modelagem de dados é fascinante.",
    "The quick brown fox jumps over the lazy dog.",
    "هذا مثال لجملة قصيرة باللغة العربية."
]
preds = pipe.predict(textos)
print(list(zip(textos, preds)))
```

> O script também imprime predições em 5 frases de exemplo ao final do `main`.

---

## 🔍 Boas práticas de robustez implementadas

- **Normalização NFKC** + *casefold* heurístico (para textos latinos)  
- **Padronização de rótulos** (ex.: “Portugese” → “Portuguese”)  
- **Split por grupos** (hash de texto normalizado) para evitar duplicatas cruzando *splits*  
- **Relatórios detalhados** + amostras de erros por par (y\_true/y\_pred)  
- **Métricas por *script*** para entender confusões entre famílias de escrita

---

## 🆘 Troubleshooting

- **Falha no download Kaggle (`kagglehub`)** → Baixe manualmente e defina **`LANGID_CSV`**.  
- **Sem `xgboost`/`gensim`** → Eles são **opcionais**; o script executa os básicos (TF‑IDF + LinearSVC/LogReg).  
- **Windows / erro de *pickle* no cache** → O cache do `Pipeline` é **desligado automaticamente** no Windows.  
- **Execução rápida**: `SKIP_EDA=1 RUN_HEAVY=0 SEARCH_SUBSAMPLE=5000 python lid22.py`.

---

## 📜 Licença

Respeita a licença e termos do **dataset** no Kaggle.
Baseado no script `lid22.py` com foco em reprodutibilidade, diagnóstico anti‑overfitting e rastreabilidade de artefatos.
