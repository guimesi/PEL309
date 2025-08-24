# -*- coding: utf-8 -*-
# =============================================================================
# LID (Language Identification) em 22 línguas: pipeline reprodutível e anti-overfitting (PEL309)
#
# O que este script faz (vista geral):
# 1) Configuração/performance: limita threads de BLAS/OMP; define SEED; habilita cache opcional (joblib.Memory).
# 2) Dados: carrega CSV via LANGID_CSV ou baixa com kagglehub; detecta automaticamente colunas de texto/rótulo.
# 3) Pré-processamento: TextNormalizer (Unicode NFKC, casefold heurístico para LATIN, remove zero-width); tokenizer simples.
# 4) Vetorização: TF-IDF (caracter/word), Count (char) e, se disponível, Word2Vec/Doc2Vec (Gensim).
# 5) Modelos: LinearSVC, LogisticRegression, MultinomialNB; opcional XGBoost (GPU via XGB_USE_GPU=1) com SVD.
# 6) Split 80/10/10 com anti-leak: estratificado por grupos (hash do texto normalizado) para impedir duplicatas entre splits.
# 7) Seleção: GridSearchCV ou HalvingGridSearchCV (quando disponível) com StratifiedKFold; salvamento de cv_results_*.
# 8) Avaliação: accuracy, macro/micro-F1, top-3 accuracy, classification_report (JSON), matrizes de confusão (bruta/normalizada).
#    Métricas agregadas por “script” (Latin, Arabic, Cyrillic, CJK, etc.) e confusão por script.
# 9) Anti-overfitting: checagem de duplicatas/"leakage" entre splits; OOF (predict out-of-fold);
#    curva de aprendizado; Y-scramble (sanidade). Tudo com CSVs/plots em artifacts/.
# 10) Artefatos: logs de execução, plots, melhores hiperparâmetros, modelo .joblib, metadata_ext.json,
#     relatórios (classification_report, oof_report, learning_curve.csv, y_scramble.csv, diag_text_leakage.json).
# 11) Inferência: imprime predições em frases exemplo ao final do run.
#
# Principais variáveis de ambiente (ligam/desligam comportamentos):
# - GS_N_JOBS: nº de processos no Grid/HalvingSearch (default = CPU count fora de notebook).
# - CV_FOLDS: nº de dobras na validação cruzada (default 3).
# - SKIP_EDA=1: pula EDA (plots e resumos).
# - RUN_HEAVY=0/1: inclui modelos extras (NB, W2V/D2V, XGB).
# - USE_HALVING=1: usa HalvingGridSearchCV (se disponível).
# - SKL_CACHE=0/1: desativa/ativa cache do Pipeline (em Windows desativa por padrão).
# - SPLIT_BY_GROUPS=0/1: ativa split por hash (anti-leak) [default ON].
# - SEARCH_SUBSAMPLE=N: amostra N exemplos para acelerar busca de hiperparâmetros.
# - CALIB_AT_END=1: calibra LinearSVC (sigmoid, cv=3) após o refit final.
# - XGB_USE_GPU=1: ativa GPU no XGBoost (tree_method=gpu_hist).
# - W2V_WORKERS: nº de workers para Word2Vec/Doc2Vec.
# - LC_POINTS, Y_SCRAMBLE_N, OOF_USE_TRVAL: controlam diagnósticos anti-overfitting.
#
# Dependências:
# - Obrigatórias: numpy, pandas, scikit-learn, matplotlib, joblib.
# - Opcionais: xgboost, gensim, kagglehub (usadas apenas se instaladas).
#
# Como executar:
# - (Opcional) export LANGID_CSV=/caminho/dataset.csv   # se não definido, tenta kagglehub
# - python seu_arquivo.py
#   → Saídas em artifacts/: logs, plots_YYYYMMDD-HHMMSS, cv_results_*.csv, modelo .joblib, metadata_ext.json, etc.
#
# Organização do código:
# - Utilidades de dados (notebook check, carregamento Kaggle/ENV, detecção de colunas)
# - Normalizador/tokenizer; Vectorizers Gensim (opcionais)
# - Mapeamento língua→script; Wrapper LabelEncodedClassifier p/ rótulos string
# - EDA helpers; Avaliação (relatórios/plots/top-k); Modelos & grids
# - Balanceamento por split; Anti-overfitting (leak, OOF, LC, Y-scramble)
# - Split estratificado por grupos (hash); MAIN (orquestra tudo)

# - **Owner:** Guilherme de Oliveira Silva
# - **Last updated:** 2025-08-24 (America/São_Paulo)
# =============================================================================

# Início do script
# ---------------------------------------------------------------------
# Setup de projeto (NLP/ML): limita threads (BLAS/OMP), carrega utilitários,
# stack científica (NumPy/Pandas/Matplotlib), scikit-learn (split/validação/
# métricas/pipeline), dependências opcionais (XGBoost, Word2Vec/Doc2Vec) e
# constantes globais (SEED, dtype, cache). Manter este bloco no topo para
# que o limite de threads seja aplicado antes de importar NumPy/sklearn.
# ---------------------------------------------------------------------
import os
for k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "1")

import re, json, time, unicodedata, random, warnings, io, sys, atexit, functools, inspect, multiprocessing, hashlib
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_predict, learning_curve
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from joblib import Memory
import joblib

# Opcionais
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from gensim.models import Word2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    HAVE_GENSIM = True
except Exception:
    HAVE_GENSIM = False

SEED = 42
VEC_DTYPE = np.float32
MEM = None  # definido em main()

# ---------------------------------------------------------------------
# Utilidades de dados (plug-and-play):
# - Detecta se está rodando em notebook (ajustes de UX/log).
# - Localiza o dataset via variável de ambiente (LANGID_CSV) ou baixa com kagglehub.
# - Carrega CSV de forma tolerante (encoding/linhas ruins).
# - Identifica automaticamente as colunas de texto e rótulo com aliases e heurísticas
#   (comprimento médio, diversidade de caracteres, cardinalidade), com fallbacks
#   para casos de 2 colunas. Usa SEED para amostragem determinística.
# ---------------------------------------------------------------------
def _running_in_notebook() -> bool:
    """Detecta se o código está rodando em um kernel IPython/Jupyter.

    Returns:
        bool: `True` se um kernel IPython ativo for detectado (ex.: Jupyter/VS Code Notebook),
        `False` caso contrário.

    Notes:
        Usa `IPython.get_ipython()` e verifica a presença de `"IPKernelApp"` no tipo do shell.
        Útil para ajustar logs/UX; não lança exceções.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in str(type(ip))
    except Exception:
        return False

def try_load_csv_from_env():
    """Obtém o caminho do dataset a partir da variável de ambiente `LANGID_CSV`.

    Returns:
        Optional[pathlib.Path]: Caminho do arquivo CSV se `LANGID_CSV` estiver definida e
        o caminho existir; caso contrário, `None`.

    Side Effects:
        Nenhum (não imprime, não lê arquivos).

    Example:
        >>> os.environ["LANGID_CSV"] = "/dados/idiomas.csv"
        >>> try_load_csv_from_env()
        PosixPath('/dados/idiomas.csv')
    """
    p = os.getenv("LANGID_CSV")
    if p and Path(p).exists():
        return Path(p)
    return None

def try_download_with_kagglehub():
    """Tenta baixar o dataset via `kagglehub` e retorna o maior CSV encontrado.

    Returns:
        Optional[pathlib.Path]: Caminho do CSV baixado/encontrado; `None` em caso de falha.

    Side Effects:
        - Pode realizar download pela internet.
        - Imprime mensagens de status/aviso no console.

    Notes:
        - Requer o pacote `kagglehub` devidamente configurado.
        - Procura recursivamente por arquivos `*.csv` no diretório retornado por
            `kagglehub.dataset_download("zarajamshaid/language-identification-datasst")`
            e escolhe o maior por tamanho de arquivo.
    """
    try:
        import kagglehub
        print("Baixando dataset via kagglehub ...")
        ds_dir = Path(kagglehub.dataset_download("zarajamshaid/language-identification-datasst"))
        csvs = list(ds_dir.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError("Nenhum CSV encontrado.")
        csvs.sort(key=lambda f: f.stat().st_size if f.exists() else 0, reverse=True)
        return csvs[0]
    except Exception as e:
        print(f"[Aviso] Falha no download com kagglehub: {e}")
        return None

def load_dataset() -> pd.DataFrame:
    """Carrega o dataset como `pandas.DataFrame`.

    Tenta primeiro `LANGID_CSV`; se não houver caminho válido, tenta baixar via `kagglehub`.

    Returns:
        pandas.DataFrame: Dados lidos do CSV.

    Raises:
        FileNotFoundError: Se não houver CSV válido (nem em `LANGID_CSV` nem via download).

    Notes:
        - Usa `encoding="utf-8"`, `engine="python"` e `on_bad_lines="skip"` (linhas ruins são ignoradas).
        - Imprime o caminho do arquivo utilizado.
    """
    csv_path = try_load_csv_from_env() or try_download_with_kagglehub()
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError("Defina LANGID_CSV ou configure Kaggle para baixar automaticamente.")
    print(f"Usando arquivo: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")

def smart_column_guess(df: pd.DataFrame):
    """Infere automaticamente as colunas de texto e rótulo em um DataFrame.

        A busca segue esta heurística em camadas:
        1) Aliases comuns: tenta casar nomes usuais para texto
            (ex.: {'text','sentence','content',...}) e para rótulo
            (ex.: {'language','label','class','target',...}).
        2) Se não encontrar, escolhe:
            - `text_col`: entre colunas de dtype 'object', a que maximiza um score de
            "semelhança com texto" = média de comprimento + 0.5 * diversidade de caracteres
            (medida em uma amostra de até 1000 linhas, usando SEED global).
            - `label_col`: entre colunas com cardinalidade entre 2 e 400 (excluindo `text_col`),
            prioriza a mais próxima de 22 categorias (idiomas) e, em seguida, a menor cardinalidade.
        3) Fallback para dataframes com exatamente duas colunas de strings:
            a coluna com maior comprimento médio vira `text_col`; a outra, `label_col`.

        Args:
            df (pandas.DataFrame): DataFrame de entrada contendo texto e rótulos.

        Returns:
            Tuple[str, str]: (`text_col`, `label_col`) — nomes das colunas inferidas.

        Raises:
            ValueError: Se não for possível identificar ambas as colunas.

        Notes:
            - Usa a constante global `SEED` para reprodutibilidade da amostra.
            - Funciona melhor quando o texto está em colunas de dtype 'object' e
            os rótulos têm cardinalidade moderada.
    """
    text_aliases = {"text","Text","sentence","content","body","paragraph"}
    label_aliases = {"language","Language","lang","label","Label","class","target"}
    text_col = label_col = None
    for c in df.columns:
        if c in text_aliases and text_col is None:
            text_col = c
        if c in label_aliases and label_col is None:
            label_col = c

    def is_textlike(s: pd.Series) -> float:
        """Calcula um score simples de "semelhança com texto" para uma coluna.

        A heurística combina duas pistas:
        1) `avg_len`: média do comprimento dos valores (após converter para string).
        2) `diversity`: número de caracteres distintos observados em uma amostra
            de até 1000 linhas (reprodutível via SEED global).
        O score final é: `score = avg_len + 0.5 * diversity`.

        Parâmetros
        ----------
        s : pandas.Series
            Série com valores possivelmente nulos; serão convertidos para `str`.

        Retorno
        -------
        float
            Valor >= 0, onde números maiores sugerem coluna mais "textual".
            Retorna `-1` em caso de erro/exceção.

        Notas
        -----
        - A amostragem usa `random_state=SEED` para reprodutibilidade.
        - Colunas de rótulos tendem a ter `avg_len` e `diversity` menores que
        colunas de texto livre; portanto, recebem score mais baixo.
    """
        try:
            s = s.dropna().astype(str)
            avg_len = s.str.len().mean()
            sample = s.sample(min(1000, len(s)), random_state=SEED)
            diversity = len(set("".join(sample.tolist())))
            return (avg_len or 0) + 0.5*diversity
        except Exception:
            return -1

    if text_col is None or label_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "O"]
        if text_col is None and obj_cols:
            scores = {c: is_textlike(df[c]) for c in obj_cols}
            text_col = max(scores, key=scores.get)
        if label_col is None:
            candidates=[]
            for c in df.columns:
                nun = df[c].nunique(dropna=True)
                if 2 <= nun <= 400:
                    candidates.append((c, nun))
            candidates = [c for c in candidates if c[0] != text_col]
            if candidates:
                candidates.sort(key=lambda x: (abs(x[1]-22), x[1]))
                label_col = candidates[0][0]

    if (text_col is None or label_col is None) and len(df.columns)==2:
        c1, c2 = list(df.columns)
        if df[c1].dtype=="O" and df[c2].dtype=="O":
            if df[c1].astype(str).str.len().mean() >= df[c2].astype(str).str.len().mean():
                text_col, label_col = c1, c2
            else:
                text_col, label_col = c2, c1

    if text_col is None or label_col is None:
        raise ValueError(f"Não foi possível identificar colunas. Colunas: {list(df.columns)}")
    return text_col, label_col

# ---------------------------------------------------------------------
# Normalização e tokenização de texto para NLP:
# - TextNormalizer (sklearn): NFKC; casefold opcional/heurístico (detecta LATIN no início);
#   remove zero-width space. Seguro para entradas nulas.
# - Tokenização simples por \w+ (Unicode) via regex.
# ---------------------------------------------------------------------
class TextNormalizer(BaseEstimator, TransformerMixin):
    """Normalizador de texto compatível com scikit-learn.

        Operações aplicadas (nesta ordem):
        1) Converte valores nulos para string vazia e força `str`.
        2) Normaliza Unicode em NFKC quando `do_normalize=True`.
        3) Case folding:
            - `lowercase=True`  → sempre aplica `casefold()`;
            - `lowercase=False` → nunca aplica;
            - `lowercase=None`  → aplica apenas se, nos 64 primeiros caracteres,
            houver algum caractere alfabético com nome Unicode contendo "LATIN".
        4) Remove U+200B (zero-width space).

        Parâmetros
        ----------
        do_normalize : bool, default=True
            Ativa a normalização Unicode NFKC.
        lowercase : Optional[bool], default=None
            Estratégia de minúsculas (True/False/None para heurística).

        Atributos
        ---------
        do_normalize : bool
            Valor fornecido no construtor.
        lowercase : Optional[bool]
            Valor fornecido no construtor.

        Notas
        -----
        - O objetivo da heurística é evitar `casefold` em scripts não-latinos.
        - Projetado para ser usado como etapa "prep" em `sklearn.Pipeline`.
    """
    def __init__(self, do_normalize=True, lowercase=None):
        self.do_normalize = do_normalize
        self.lowercase = lowercase
    def fit(self, X, y=None):
        """Compatibilidade com `TransformerMixin` (não aprende parâmetros)."""
        return self
    def transform(self, X):
        """Aplica a sequência de normalizações a um iterável de textos.

        Parâmetros
        ----------
        X : Iterable[Any]
            Sequência de valores textuais (serão convertidos para `str`).

        Retorno
        -------
        List[str]
            Lista de textos normalizados.
        """
        out=[]
        for s in X:
            s = "" if pd.isna(s) else str(s)
            if self.do_normalize:
                s = unicodedata.normalize("NFKC", s)
            if self.lowercase is True:
                s = s.casefold()
            elif self.lowercase is None:
                head = s[:64]
                if any("LATIN" in unicodedata.name(ch, "") for ch in head if ch.isalpha()):
                    s = s.casefold()
            s = s.replace("\u200b","")
            out.append(s)
        return out

# Regex de palavras: \w+ com UNICODE (letras/dígitos/_ em múltiplos scripts).
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
def simple_word_tokenize(text: str):
    """Tokenizador simples baseado em regex `\\w+` (compatível com Unicode).

    Converte entradas não-string para `str` (ou "" se NaN) e retorna a lista
    de tokens encontrados pela expressão `\\w+`.

    Parâmetros
    ----------
    text : str
        Texto de entrada.

    Retorno
    -------
    List[str]
        Lista de tokens (letras/dígitos/sublinhado). Não remove stopwords,
        não aplica stemming/lemmatização.

    Observação
    ----------
    Como usa `\\w+`, números e sublinhados são preservados, e scripts não-latinos
    (ex.: árabe, cirílico) também são suportados.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return TOKEN_RE.findall(text)

# ---------------------------------------------------------------------
# Embeddings clássicos opcionais (Gensim):
# - W2VVectorizer: treina Word2Vec no corpus e representa cada documento
#   pela média dos vetores das palavras conhecidas.
# - D2VVectorizer: treina Doc2Vec e usa infer_vector para obter o vetor
#   de cada documento. Controla paralelismo via W2V_WORKERS e usa SEED.
# ---------------------------------------------------------------------
if HAVE_GENSIM:
    class W2VVectorizer(BaseEstimator, TransformerMixin):
        """Vetorizador baseado em Word2Vec treinado no próprio corpus.

        Cada documento é representado pela média dos vetores das palavras conhecidas
        no vocabulário do modelo (`model_.wv`). Palavras OOV são ignoradas; se um
        documento não contiver nenhuma palavra conhecida, retorna um vetor zero.

        Parâmetros
        ----------
        size : int, default=200
            Dimensão do embedding (parâmetro `vector_size` do Word2Vec).
        window : int, default=5
            Tamanho da janela de contexto.
        min_count : int, default=2
            Frequência mínima para uma palavra entrar no vocabulário.
        sg : int, default=1
            Arquitetura: 0=CBOW, 1=Skip-gram.
        epochs : int, default=5
            Número de épocas de treino.
        workers : Optional[int], default=None
            Número de processos de treino. Se None, usa `int(os.getenv("W2V_WORKERS","1"))`.

        Atributos
        ---------
        model_ : gensim.models.Word2Vec
            Modelo treinado no `fit`.
        """
        def __init__(self, size=200, window=5, min_count=2, sg=1, epochs=5, workers=None):
            self.size=size; self.window=window; self.min_count=min_count
            self.sg=sg; self.epochs=epochs
            self.workers = int(os.getenv("W2V_WORKERS", "1")) if workers is None else workers
            self.model_=None
        def fit(self, X, y=None):
            """Treina o Word2Vec sobre os textos.

            Parâmetros
            ----------
            X : Iterable[str]
                Coleção de documentos (strings).
            y : Ignorado
                Presente por compatibilidade sklearn.

            Retorno
            -------
            self : W2VVectorizer
            """
            sentences=[simple_word_tokenize(t) for t in X]
            from gensim.models import Word2Vec
            self.model_=Word2Vec(
                sentences=sentences, vector_size=self.size, window=self.window,
                min_count=self.min_count, sg=self.sg, epochs=self.epochs,
                workers=self.workers, seed=SEED
            )
            return self
        def transform(self, X):
            """Gera embeddings por documento via média dos vetores de palavras.

            Parâmetros
            ----------
            X : Iterable[str]
                Coleção de documentos (strings).

            Retorno
            -------
            numpy.ndarray
                Matriz (n_docs, size) em float32.
            """
            def docvec(tokens):
                vecs=[self.model_.wv[w] for w in tokens if w in self.model_.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(self.size, dtype=np.float32)
            return np.vstack([docvec(simple_word_tokenize(t)) for t in X])

    class D2VVectorizer(BaseEstimator, TransformerMixin):
        """Vetorizador baseado em Doc2Vec (Gensim).

        Treina um modelo Doc2Vec no corpus e, na transformação, usa `infer_vector`
        para obter o embedding de cada documento.

        Parâmetros
        ----------
        size : int, default=200
            Dimensão do vetor de documento.
        window : int, default=5
            Tamanho de janela.
        min_count : int, default=2
            Frequência mínima para vocabulário.
        dm : int, default=1
            Arquitetura: 1=PV-DM, 0=PV-DBOW.
        epochs : int, default=20
            Épocas de treino.
        workers : Optional[int], default=None
            Nº de processos (ou `int(os.getenv("W2V_WORKERS","1"))` se None).

        Atributos
        ---------
        model_ : gensim.models.Doc2Vec
            Modelo treinado no `fit`.
        """
        def __init__(self, size=200, window=5, min_count=2, dm=1, epochs=20, workers=None):
            self.size=size; self.window=window; self.min_count=min_count
            self.dm=dm; self.epochs=epochs
            self.workers = int(os.getenv("W2V_WORKERS", "1")) if workers is None else workers
            self.model_=None
        def fit(self, X, y=None):
            """Treina o Doc2Vec sobre os textos tokenizados.

            Parâmetros
            ----------
            X : Iterable[str]
                Coleção de documentos (strings).
            y : Ignorado
                Presente por compatibilidade sklearn.

            Retorno
            -------
            self : D2VVectorizer
            """
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            docs=[TaggedDocument(words=simple_word_tokenize(t), tags=[i]) for i,t in enumerate(X)]
            self.model_=Doc2Vec(
                vector_size=self.size, window=self.window, min_count=self.min_count,
                dm=self.dm, workers=self.workers, epochs=self.epochs, seed=SEED
            )
            self.model_.build_vocab(docs); self.model_.train(docs, total_examples=len(docs), epochs=self.epochs)
            return self
        def transform(self, X):
            """Infere o vetor de documento para cada texto.

            Parâmetros
            ----------
            X : Iterable[str]
                Coleção de documentos (strings).

            Retorno
            -------
            numpy.ndarray
                Matriz (n_docs, size) em float32.
            """
            return np.vstack([self.model_.infer_vector(simple_word_tokenize(t)) for t in X])

# ---------------------------------------------------------------------
# Mapeamento de sistemas de escrita (scripts):
# - Dicionário língua→script (Latin, Arabic, Cyrillic, CJK, etc.).
# - Funções auxiliares para derivar o script a partir dos rótulos e
#   calcular métricas de desempenho agregadas por script (relatório e CM).
# ---------------------------------------------------------------------
LANG_TO_SCRIPT = {
    "Arabic":"Arabic","Persian":"Arabic","Urdu":"Arabic","Pushto":"Arabic","Pashto":"Arabic",
    "Hindi":"Devanagari","Russian":"Cyrillic","Chinese":"CJK","Japanese":"CJK","Korean":"Hangul",
    "Thai":"Thai","Tamil":"Tamil","English":"Latin","Dutch":"Latin","Estonian":"Latin","French":"Latin",
    "Indonesian":"Latin","Latin":"Latin","Portuguese":"Latin","Portugese":"Latin","Romanian":"Latin",
    "Spanish":"Latin","Swedish":"Latin","Turkish":"Latin","Italian":"Latin","German":"Latin",
    "Polish":"Latin","Czech":"Latin","Norwegian":"Latin","Danish":"Latin","Finnish":"Latin","Hungarian":"Latin",
}
def scripts_from_labels(labels):
    """Mapeia rótulos de língua para sistemas de escrita (scripts).

    Parâmetros
    ----------
    labels : Iterable[str] | numpy.ndarray
        Sequência de rótulos de idioma (e.g., "Portuguese", "Arabic").

    Retorno
    -------
    numpy.ndarray
        Array de mesmos comprimento/ordem com os scripts correspondentes
        (e.g., "Latin", "Arabic"). Rótulos não mapeados viram "Unknown".

    Notas
    -----
    - A correspondência depende de `LANG_TO_SCRIPT`. Se necessário, normalize
      os rótulos antes (capitalização/aliases).
    """
    return np.array([LANG_TO_SCRIPT.get(lbl,"Unknown") for lbl in labels])

def script_level_metrics(y_true_lang, y_pred_lang):
    """Calcula métricas agregadas por sistema de escrita (script).

    Converte rótulos de idioma predito/verdadeiro em scripts e então computa
    um `classification_report` e uma matriz de confusão entre scripts.

    Parâmetros
    ----------
    y_true_lang : Iterable[str] | numpy.ndarray
        Rótulos verdadeiros de idioma por amostra.
    y_pred_lang : Iterable[str] | numpy.ndarray
        Rótulos previstos de idioma por amostra.

    Retorno
    -------
    scripts : List[str]
        Lista ordenada de scripts considerados (presentes em `y_true_lang`).
    rep : Dict[str, Dict[str, float]]
        Dicionário no formato do `sklearn.metrics.classification_report(output_dict=True)`,
        contendo precisão, recall, f1-score e suporte por script.
    cm : numpy.ndarray
        Matriz de confusão (shape = [n_scripts, n_scripts]) na ordem de `scripts`.

    Notas
    -----
    - `zero_division=0` evita warnings quando não há positivos para um script.
    - Útil para analisar confusões de alto nível (ex.: Latin vs. Cyrillic vs. CJK).
    """
    true_s = scripts_from_labels(y_true_lang); pred_s = scripts_from_labels(y_pred_lang)
    scripts = sorted(np.unique(true_s))
    rep = classification_report(true_s, pred_s, labels=scripts, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_s, pred_s, labels=scripts)
    return scripts, rep, cm

# ---------------------------------------------------------------------
# Wrapper para estimadores (XGB) que exigem rótulos numéricos:
# - Aceita y como strings, faz LabelEncoder interno e mantém classes_ originais.
# - Propaga/recebe hiperparâmetros via prefixo "estimator__" (compatível com GridSearchCV/Pipeline).
# - Ajusta automaticamente "num_class" quando disponível (ex.: XGBoost).
# - Suporta predict() com des-encoding e delega predict_proba() quando o estimador interno tiver.
# ---------------------------------------------------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper para estimadores scikit-learn que exigem rótulos numéricos.

    Objetivo
    --------
    Permitir treinar/avaliar modelos que esperam `y` como inteiros usando
    rótulos de classe em string. Internamente, aplica `LabelEncoder`,
    ajusta o estimador clonado e expõe `classes_` originais.

    Recursos
    --------
    - Compatível com `Pipeline`/`GridSearchCV`: hiperparâmetros do estimador
        interno são acessados via prefixo `estimator__param`.
    - Se o estimador possuir o parâmetro `num_class` (ex.: XGBoost), ele é
        configurado automaticamente com o número de classes.

    Atributos
    ---------
    estimator : BaseEstimator
        Estimador base (não ajustado) fornecido no construtor.
    estimator_ : BaseEstimator
        Cópia ajustada do estimador após `fit`.
    _le : sklearn.preprocessing.LabelEncoder
        Codificador de rótulos usado internamente.
    classes_ : numpy.ndarray
        Rótulos originais (strings) na ordem do `LabelEncoder`.
    """
    def __init__(self, estimator):
        """Inicializa o wrapper.

        Parâmetros
        ----------
        estimator : BaseEstimator
            Estimador scikit-learn a ser envolvido (ex.: XGBClassifier, SVC, etc.).
        """
        self.estimator = estimator
    def get_params(self, deep=True):
        """Retorna hiperparâmetros no formato compatível com scikit-learn.

        Inclui:
        - A própria chave `"estimator"`.
        - Parâmetros do estimador interno com prefixo `estimator__`.

        Parâmetros
        ----------
        deep : bool, default=True
            Se `True`, inclui parâmetros aninhados do estimador interno.

        Retorno
        -------
        dict
            Dicionário de parâmetros expandidos.
        """
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=deep).items():
                params[f"estimator__{k}"] = v
        return params
    def set_params(self, **params):
        """Define hiperparâmetros no wrapper e/ou no estimador interno.

        Regras
        ------
        - Chaves com prefixo `estimator__` são repassadas ao estimador interno.
        - Demais chaves são atribuídas ao próprio wrapper.

        Parâmetros
        ----------
        **params
            Parâmetros a definir.

        Retorno
        -------
        LabelEncodedClassifier
            A própria instância (para encadeamento).
        """
        est_params, rest = {}, {}
        for k, v in params.items():
            if k.startswith("estimator__"):
                est_params[k.split("__", 1)[1]] = v
            else:
                rest[k] = v
        if est_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_params)
        for k, v in rest.items():
            setattr(self, k, v)
        return self
    def fit(self, X, y):
        """Ajusta o `LabelEncoder`, prepara o estimador e treina com rótulos codificados.

        Passos
        ------
        1) `LabelEncoder.fit_transform(y)` → `y_enc`.
        2) `clone(self.estimator)` → `self.estimator_`.
        3) Se suportado, define `num_class` = `len(classes_)`.
        4) `self.estimator_.fit(X, y_enc)`.

        Parâmetros
        ----------
        X : array-like ou matriz esparsa (n_amostras, n_features)
            Atributos de treino.
        y : array-like (n_amostras,)
            Rótulos (strings ou quaisquer hashables).

        Retorno
        -------
        LabelEncodedClassifier
            A própria instância ajustada.

        Atributos definidos
        -------------------
        classes_ : numpy.ndarray
            Rótulos originais na ordem interna do modelo.
        estimator_ : BaseEstimator
            Estimador treinado com rótulos numéricos.
        """
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        try:
            if "num_class" in self.estimator_.get_params():
                self.estimator_.set_params(num_class=len(self._le.classes_))
        except Exception:
            pass
        self.estimator_.fit(X, y_enc)
        self.classes_ = self._le.classes_
        return self
    def predict(self, X):
        """Prevê rótulos originais (desfazendo a codificação).

        Parâmetros
        ----------
        X : array-like ou matriz esparsa (n_amostras, n_features)
            Atributos para predição.

        Retorno
        -------
        numpy.ndarray
            Rótulos previstos no espaço original (strings).

        Notas
        -----
        - Converte a saída do estimador para `int` antes do `inverse_transform`.
        """
        y_enc = self.estimator_.predict(X)
        y_enc = np.asarray(y_enc, dtype=int)
        return self._le.inverse_transform(y_enc)
    def predict_proba(self, X):
        """Retorna probabilidades de classe do estimador interno (se suportado).

        Parâmetros
        ----------
        X : array-like ou matriz esparsa (n_amostras, n_features)
            Atributos para predição de probabilidade.

        Retorno
        -------
        numpy.ndarray
            Probabilidades com shape (n_amostras, n_classes) na ordem de `classes_`.

        Raises
        ------
        AttributeError
            Se o estimador interno não implementar `predict_proba`.
        """
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)
        raise AttributeError("Inner estimator não suporta predict_proba")

# ---------------------------------------------------------------------
# EDA helpers:
# - Gráficos de balanceamento por classe e distribuições de comprimento.
# - Boxplot por língua e histograma global.
# - Lista de n-gramas de caracteres mais frequentes por língua (amostra).
# - Usa plot_saver.savefig(fig, <nome>) para persistir PNGs e SEED p/ reprodutibilidade.
# ---------------------------------------------------------------------
def plot_class_distribution(counts, plot_saver):
    """Plota e salva a distribuição de classes (contagem por língua).

    Parâmetros
    ----------
    counts : pandas.Series
        Série indexada por rótulo de língua com as contagens de exemplos.
    plot_saver : object
        Objeto com método `savefig(fig, name, dpi=...)` (ex.: PlotSaver)
        usado para persistir a figura em PNG.

    Efeitos colaterais
    ------------------
    - Exibe o gráfico em tela (`plt.show()`).
    - Salva a figura como "eda_class_distribution.png" via `plot_saver`.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    counts.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Contagem por língua (balanceamento)"); ax.set_ylabel("n"); plt.tight_layout()
    plot_saver.savefig(fig, "eda_class_distribution.png"); plt.show()

def plot_length_distributions(df, text_col, label_col, plot_saver):
    """Gera EDA de comprimentos de texto (global e por língua) e salva os plots.

    Produz:
        1) Histograma global de comprimento em caracteres.
        2) Boxplot de comprimento por língua (sem outliers).

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo colunas de texto e rótulo.
    text_col : str
        Nome da coluna de texto.
    label_col : str
        Nome da coluna de rótulo (língua).
    plot_saver : object
        Objeto com método `savefig(fig, name, dpi=...)` (ex.: PlotSaver).

    Efeitos colaterais
    ------------------
    - Imprime `describe()` da coluna de comprimentos.
    - Exibe os gráficos em tela (`plt.show()`).
    - Salva PNGs: "eda_length_hist_global.png" e
        "eda_length_boxplot_by_language.png".
    """
    d = df.copy(); d["len"]=d[text_col].astype(str).str.len()
    print("\nResumo global de comprimentos (caracteres):"); print(d["len"].describe())
    fig1, ax1 = plt.subplots(figsize=(10,5))
    d["len"].hist(bins=60, ax=ax1); ax1.set_title("Comprimento (caracteres) - global")
    ax1.set_xlabel("tamanho"); ax1.set_ylabel("freq"); plt.tight_layout()
    plot_saver.savefig(fig1, "eda_length_hist_global.png"); plt.show()
    order = d.groupby(label_col)["len"].median().sort_values().index.tolist()
    data = [d.loc[d[label_col]==L, "len"].values for L in order]
    fig2, ax2 = plt.subplots(figsize=(14,7))
    ax2.boxplot(data, showfliers=False)
    ax2.set_xticks(range(1, len(order)+1)); ax2.set_xticklabels(order, rotation=90)
    ax2.set_title("Boxplot de comprimento por língua"); ax2.set_ylabel("caracteres")
    plt.tight_layout()
    plot_saver.savefig(fig2, "eda_length_boxplot_by_language.png"); plt.show()

def top_char_ngrams_per_language(df, text_col, label_col, n=3, topk=10):
    """Imprime os n-gramas de caracteres mais frequentes por língua (amostra).

    Seleciona as 6 línguas com mais exemplos, amostra até 400 textos por língua,
    computa n-gramas de caracteres de tamanho `n` e lista os `topk` mais comuns.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame com textos e rótulos.
    text_col : str
        Nome da coluna de texto.
    label_col : str
        Nome da coluna de rótulo (língua).
    n : int, default=3
        Tamanho do n-grama de caracteres.
    topk : int, default=10
        Quantidade de n-gramas a exibir por língua.

    Efeitos colaterais
    ------------------
    - Imprime no console os `topk` n-gramas para cada língua selecionada.
    """
    langs = df[label_col].value_counts().head(6).index.tolist()
    def char_ngrams(s, n): s = str(s); return [s[i:i+n] for i in range(len(s)-n+1)]
    for L in langs:
        sub = df[df[label_col]==L][text_col].astype(str)
        c = Counter()
        for t in sub.sample(min(400, len(sub)), random_state=SEED):
            c.update(char_ngrams(t, n))
        print(f"\nTop {topk} char {n}-grams para [{L}]:")
        for gram, freq in c.most_common(topk): print(f"{gram!r}: {freq}")

# ---------------------------------------------------------------------
# Avaliação:
# - get_score_matrix: extrai matriz de scores (proba/decision_function/one-hot).
# - topk_accuracy: calcula acurácia Top-k.
# - evaluate_on_split: avalia um pipeline no split (accuracy/F1, relatório),
#   plota/salva matrizes de confusão (bruta/normalizada), métricas por script,
#   top-3 accuracy e exemplos de erros. Usa ARTIFACTS_DIR e plot_saver.
# ---------------------------------------------------------------------
def get_score_matrix(fitted_pipe: Pipeline, X_data):
    """Gera uma matriz de scores alinhada às classes do classificador do Pipeline.

    Estratégia:
        1) Se o passo "clf" tiver `predict_proba`, retorna `pipeline.predict_proba(X_data)`.
        2) Senão, se tiver `decision_function`, usa `pipeline.decision_function(X_data)` e,
            no caso binário 1D, converte para 2 colunas como `[-score, score]`.
        3) Caso nenhum dos dois exista, faz fallback para uma matriz one-hot a partir de `predict()`.

    Parâmetros
    ----------
    fitted_pipe : sklearn.pipeline.Pipeline
        Pipeline já ajustado contendo um passo "clf" com atributo `classes_`.
    X_data : array-like ou matriz esparsa
        Atributos a serem pontuados.

    Retorno
    -------
    numpy.ndarray
        Matriz de forma (n_amostras, n_classes) com scores por classe.
        A ordem das colunas segue `fitted_pipe.named_steps["clf"].classes_`.

    Notas
    -----
    - Útil para calcular métricas baseadas em ranking (ex.: Top-k).
    - Em classificadores lineares sem `predict_proba` (ex.: `LinearSVC`), a via
        `decision_function` oferece scores ordenáveis; se ausente, cai no one-hot.
    """
    clf = fitted_pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"): return fitted_pipe.predict_proba(X_data)
    elif hasattr(clf, "decision_function"):
        scores = fitted_pipe.decision_function(X_data)
        if scores.ndim == 1: scores = np.vstack([-scores, scores]).T
        return scores
    else:
        preds = fitted_pipe.predict(X_data)
        classes = clf.classes_; mat = np.zeros((len(preds), len(classes)), dtype=float)
        col = {c:i for i,c in enumerate(classes)}
        for i,p in enumerate(preds): mat[i, col[p]] = 1.0
        return mat

def topk_accuracy(y_true, score_mat, classes, k=3):
    """Calcula Top-k accuracy a partir de uma matriz de scores.

    Definição:
        Uma previsão para a i-ésima amostra é considerada correta se o rótulo
        verdadeiro estiver entre as `k` classes com maiores scores na linha `i`.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos verdadeiros (mesmo domínio de `classes`).
    score_mat : numpy.ndarray
        Matriz (n_amostras, n_classes) com scores por classe.
    classes : array-like
        Sequência de rótulos na MESMA ordem das colunas de `score_mat`.
    k : int, default=3
        Quantidade de classes no topo a considerar.

    Retorno
    -------
    float
        Top-k accuracy no intervalo [0, 1].

    Observações
    -----------
    - A ordenação usa `argsort` decrescente por linha.
    - Se houver empates de score, o comportamento segue a ordem retornada por `argsort`.
    """
    hits = 0
    for i, yt in enumerate(y_true):
        row = score_mat[i]; topk_idx = np.argsort(row)[::-1][:k]
        if yt in [classes[j] for j in topk_idx]: hits += 1
    return hits / len(y_true)

def evaluate_on_split(pipe, X_test, y_test, title="TEST", exp_name="experiment", save_prefix=None, plot_saver=None):
    """Avalia um Pipeline em um split (ex.: TEST) e salva artefatos de métricas/plots.

    Fluxo:
        1) Prediz rótulos e imprime Accuracy, Macro-F1, Micro-F1.
        2) Gera `classification_report` (imprime e salva JSON em ARTIFACTS_DIR).
        3) Plota/salva matriz de confusão bruta e normalizada (se `plot_saver` fornecido).
        4) Calcula Top-3 accuracy usando `get_score_matrix`.
        5) Agrega métricas por sistema de escrita (script) e plota confusão por script.
        6) Salva exemplos de erros por par (y_true, y_pred) e lista os pares mais frequentes.

    Parâmetros
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline AJUSTADO contendo passo "clf" com `classes_`.
    X_test : array-like
        Atributos do split de avaliação.
    y_test : array-like
        Rótulos verdadeiros do split.
    title : str, default="TEST"
        Rótulo textual para prints/títulos dos gráficos.
    exp_name : str, default="experiment"
        Nome do experimento (usado em nomes de arquivos).
    save_prefix : Optional[str], default=None
        Prefixo para arquivos salvos; se None, usa `exp_name`.
    plot_saver : Optional[object], default=None
        Objeto com método `savefig(fig, name, dpi=...)`. Se None, não salva/gera plots.

    Retorno
    -------
    dict
        Dicionário com principais métricas e caminhos de artefatos:
        {
            "accuracy": float,
            "macro_f1": float,
            "micro_f1": float,
            "top3_acc": float,
            "confusion_matrix": List[List[int]],
            "classes": List[str],
            "script_metrics": Dict[str, Any],
            "script_confusion": List[List[int]],
            "classification_report_path": str,
            "errors_csv_path": Optional[str],
        }

    Efeitos colaterais
    ------------------
    - Requer a variável global `ARTIFACTS_DIR` para salvar arquivos.
    - Imprime métricas no console.
    - Salva JSON do classification_report e, se aplicável, PNGs das matrizes de confusão.
    - Salva CSV com exemplos de erros (até 3 por par confuso).

    Dependências
    ------------
    - Usa `script_level_metrics` e `get_score_matrix` definidos no módulo.
    - Usa `matplotlib` para os gráficos quando `plot_saver` é fornecido.
    """
    if save_prefix is None: save_prefix = exp_name
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred); f1m = f1_score(y_test, y_pred, average="macro"); f1mi = f1_score(y_test, y_pred, average="micro")
    print(f"\n{title} | accuracy={acc:.4f} | macroF1={f1m:.4f} | microF1={f1mi:.4f}\n")

    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("Classification report:"); print(classification_report(y_test, y_pred, zero_division=0))
    cls_report_path = ARTIFACTS_DIR / f"classification_report_{title.lower()}_{save_prefix}.json"
    with open(cls_report_path, "w", encoding="utf-8") as f: json.dump(cls_report, f, ensure_ascii=False, indent=2)
    print(f"[i] Classification report salvo em: {cls_report_path.resolve()}")

    classes = pipe.named_steps["clf"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    if plot_saver is not None:
        fig, ax = plt.subplots(figsize=(10,10))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)
        ax.set_title(f"Matriz de confusão - {title}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="black")
        plot_saver.savefig(fig, f"confusion_matrix_{title.lower()}_{save_prefix}.png"); plt.tight_layout(); plt.show()

        cm_norm = cm.astype(np.float64); row_sums = cm_norm.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1.0
        cm_norm = cm_norm / row_sums
        fign, axn = plt.subplots(figsize=(10,10))
        imn = axn.imshow(cm_norm, cmap="Blues")
        axn.set_xticks(range(len(classes))); axn.set_xticklabels(classes, rotation=90)
        axn.set_yticks(range(len(classes))); axn.set_yticklabels(classes)
        axn.set_title(f"Matriz de confusão normalizada - {title}")
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]): axn.text(j, i, f"{100*cm_norm[i, j]:.1f}%", ha="center", va="center", fontsize=7)
        plt.colorbar(imn, fraction=0.046, pad=0.04)
        plot_saver.savefig(fign, f"confusion_matrix_norm_{title.lower()}_{save_prefix}.png"); plt.tight_layout(); plt.show()

    scores = get_score_matrix(pipe, X_test); top3 = topk_accuracy(y_test, scores, classes, k=3)
    print(f"Top-3 accuracy ({title}): {top3:.4f}")

    scripts, rep_s, cm_s = script_level_metrics(y_test, y_pred)
    print("\nMétricas agregadas por SCRIPT:")
    for s in scripts:
        r = rep_s.get(s, {})
        if r: print(f"- {s}: precision={r.get('precision',0):.3f} | recall={r.get('recall',0):.3f} | f1={r.get('f1-score',0):.3f} | support={r.get('support',0)}")

    if plot_saver is not None:
        fig2, ax2 = plt.subplots(figsize=(6,5))
        im = ax2.imshow(cm_s, cmap="Blues")
        ax2.set_xticks(range(len(scripts))); ax2.set_xticklabels(scripts, rotation=45, ha="right")
        ax2.set_yticks(range(len(scripts))); ax2.set_yticklabels(scripts)
        ax2.set_title(f"Confusão por SCRIPT - {title}")
        for i in range(cm_s.shape[0]):
            for j in range(cm_s.shape[1]): ax2.text(j, i, cm_s[i, j], ha="center", va="center", fontsize=9)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plot_saver.savefig(fig2, f"script_confusion_{title.lower()}_{save_prefix}.png"); plt.tight_layout(); plt.show()

    err_df = pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred})
    err_only = err_df[err_df["y_true"] != err_df["y_pred"]]
    errors_csv_path = None
    if len(err_only) > 0:
        examples_err = (err_only.groupby(["y_true","y_pred"], as_index=False, group_keys=False).apply(lambda g: g.head(3))).reset_index(drop=True)
        errors_csv_path = ARTIFACTS_DIR / f"errors_{title.lower()}_{save_prefix}.csv"
        examples_err.to_csv(errors_csv_path, index=False, encoding="utf-8")
        pairs = (err_only.groupby(["y_true","y_pred"]).size().reset_index(name="count").sort_values("count", ascending=False))
        print("\nPares mais confundidos (top 10):"); print(pairs.head(10))
        print(f"[i] Exemplos de erros salvos em: {errors_csv_path.resolve()}")

    return {
        "accuracy": float(acc), "macro_f1": float(f1m), "micro_f1": float(f1mi),
        "top3_acc": float(top3), "confusion_matrix": cm.tolist(),
        "classes": list(classes), "script_metrics": rep_s, "script_confusion": cm_s.tolist(),
        "classification_report_path": str(cls_report_path),
        "errors_csv_path": (str(errors_csv_path) if errors_csv_path else None),
    }

# ---------------------------------------------------------------------
# Modelos e grades de hiperparâmetros:
# - Pipelines prontos (TF-IDF char/word, Count char) com SVM Linear,
#   Regressão Logística e Naive Bayes.
# - Versões opcionais com Gensim (Word2Vec/Doc2Vec + LogReg).
# - Variante com XGBoost sobre TF-IDF reduzido via SVD (GPU opcional: XGB_USE_GPU=1).
# - Funções grid_* definem espaços de busca para GridSearchCV/Pipeline.
# ---------------------------------------------------------------------
def make_tfidf_char_linsvm():
    """Cria um Pipeline TF-IDF (caracteres) + LinearSVC.

    Etapas:
        - prep: TextNormalizer (NFKC e casefold heurístico).
        - vec : TfidfVectorizer em n-gramas de caracteres (2–5), com sublinear_tf e L2.
        - clf : LinearSVC (dual='auto').

    Retorno
    -------
    sklearn.pipeline.Pipeline
        Pipeline configurado, com `memory=MEM`.
    """
    return Pipeline([
        ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
        ("vec", TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3, max_df=0.995,
                                max_features=150_000, sublinear_tf=True, norm="l2", dtype=VEC_DTYPE)),
        ("clf", LinearSVC(dual="auto"))
    ], memory=MEM)

def grid_tfidf_char_linsvm():
    """Espaço de busca de hiperparâmetros para TF-IDF(char)+LinearSVC.

    Retorno
    -------
    dict
        Dicionário no formato aceito por GridSearchCV.
    """
    return {"prep__do_normalize":[True],
            "prep__lowercase":[None],
            "vec__ngram_range":[(2,5)],
            "vec__min_df":[3],
            "vec__max_features":[150_000],
            "clf__C":[0.5,1.0,2.0]}

def make_tfidf_word_logreg():
    """Cria um Pipeline TF-IDF (palavras) + Regressão Logística multinomial.

    Etapas:
        - prep: TextNormalizer.
        - vec : TfidfVectorizer em 1–2-gramas de palavra.
        - clf : LogisticRegression (solver='saga', multi_class='multinomial').

    Retorno
    -------
    sklearn.pipeline.Pipeline
        Pipeline configurado, com `memory=MEM`.
    """
    return Pipeline([
        ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
        ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.995,
                                max_features=200_000, dtype=VEC_DTYPE)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, solver="saga", multi_class="multinomial", random_state=SEED))
    ], memory=MEM)

def grid_tfidf_word_logreg():
    """Espaço de busca de hiperparâmetros para TF-IDF(word)+LogReg.

    Retorno
    -------
    dict
        Dicionário no formato aceito por GridSearchCV.
    """
    return {"prep__do_normalize":[True],
            "prep__lowercase":[None,True],
            "vec__ngram_range":[(1,1),(1,2)],
            "vec__min_df":[3],
            "vec__max_features":[200_000],
            "clf__C":[0.5,1.0,2.0]}

def make_count_char_nb():
    """Cria um Pipeline CountVectorizer (caracteres) + MultinomialNB.

    Etapas:
        - prep: TextNormalizer.
        - vec : CountVectorizer em n-gramas de caracteres (2–5).
        - clf : MultinomialNB.

    Retorno
    -------
    sklearn.pipeline.Pipeline
        Pipeline configurado, com `memory=MEM`.
    """
    from sklearn.naive_bayes import MultinomialNB
    return Pipeline([
        ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
        ("vec", CountVectorizer(analyzer="char", ngram_range=(2,5), min_df=2, max_df=0.995,
                                max_features=150_000, dtype=VEC_DTYPE)),
        ("clf", MultinomialNB())
    ], memory=MEM)

def grid_count_char_nb():
    """Espaço de busca de hiperparâmetros para Count(char)+MultinomialNB.

    Retorno
    -------
    dict
        Dicionário no formato aceito por GridSearchCV.
    """
    return {"prep__do_normalize":[True],
            "prep__lowercase":[None],
            "vec__ngram_range":[(2,5),(3,6)],
            "vec__min_df":[2,5],
            "vec__max_features":[120_000,150_000],
            "clf__alpha":[0.5,1.0,2.0]}

if HAVE_GENSIM:
    def make_w2v_logreg():
        """Cria um Pipeline Word2Vec(média de vetores) + Regressão Logística.

        Etapas:
            - prep: TextNormalizer.
            - w2v : W2VVectorizer (treinado no próprio corpus).
            - clf : LogisticRegression multinomial.

        Retorno
        -------
        sklearn.pipeline.Pipeline
            Pipeline configurado, com `memory=MEM`.
        """
        return Pipeline([
            ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
            ("w2v", W2VVectorizer(size=200, window=5, min_count=2, sg=1, epochs=5)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, solver="lbfgs", multi_class="multinomial", random_state=SEED))
        ], memory=MEM)
    def grid_w2v_logreg():
        """Espaço de busca de hiperparâmetros para W2V+LogReg.

        Retorno
        -------
        dict
            Dicionário no formato aceito por GridSearchCV.
        """
        return {"prep__do_normalize":[True],
                "prep__lowercase":[None],
                "w2v__size":[150,200],
                "w2v__window":[5],
                "clf__C":[1.0,2.0]}

    def make_d2v_logreg():
        """Cria um Pipeline Doc2Vec(infer_vector) + Regressão Logística.

        Etapas:
            - prep: TextNormalizer.
            - d2v : D2VVectorizer (treinado no próprio corpus).
            - clf : LogisticRegression multinomial.

        Retorno
        -------
        sklearn.pipeline.Pipeline
            Pipeline configurado, com `memory=MEM`.
        """
        return Pipeline([
            ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
            ("d2v", D2VVectorizer(size=200, window=5, min_count=2, dm=1, epochs=20)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, solver="lbfgs", multi_class="multinomial", random_state=SEED))
        ], memory=MEM)
    def grid_d2v_logreg():
        """Espaço de busca de hiperparâmetros para D2V+LogReg.

        Retorno
        -------
        dict
            Dicionário no formato aceito por GridSearchCV.
        """
        return {"prep__do_normalize":[True],
                "prep__lowercase":[None],
                "d2v__size":[150,200],
                "d2v__window":[5],
                "clf__C":[1.0,2.0]}

def make_xgb_tfidf_word():
    """Cria um Pipeline TF-IDF(word) → SVD → XGBoost (multiclasse).

    Observações:
        - Requer `xgboost` instalado; retorna `None` se `HAVE_XGB=False`.
        - GPU opcional via `XGB_USE_GPU=1` (tree_method='gpu_hist', predictor='gpu_predictor').
        - Usa `LabelEncodedClassifier` para aceitar rótulos em string.

    Etapas:
        - prep: TextNormalizer.
        - vec : TfidfVectorizer (1–2-gramas de palavra).
        - svd : TruncatedSVD (default 256 componentes) para compressão.
        - clf : XGBClassifier (objective='multi:softprob') dentro do wrapper.

    Retorno
    -------
    Optional[sklearn.pipeline.Pipeline]
        Pipeline configurado (ou `None` se XGBoost indisponível).
    """
    if not HAVE_XGB: return None
    use_gpu = os.getenv("XGB_USE_GPU", "0") == "1"
    tree_method = "gpu_hist" if use_gpu else "hist"
    predictor = "gpu_predictor" if use_gpu else "auto"
    xgb = XGBClassifier(objective="multi:softprob",
                        n_estimators=250, max_depth=6,
                        learning_rate=0.2, subsample=0.8, colsample_bytree=0.8,
                        tree_method=tree_method, predictor=predictor,
                        eval_metric="mlogloss", verbosity=0, random_state=SEED, n_jobs=1)
    return Pipeline([
        ("prep", TextNormalizer(do_normalize=True, lowercase=None)),
        ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.995,
                                max_features=250_000, dtype=VEC_DTYPE)),
        ("svd", TruncatedSVD(n_components=256, random_state=SEED)),
        ("clf", LabelEncodedClassifier(xgb))
    ], memory=MEM)

def grid_xgb_tfidf_word():
    """Espaço de busca de hiperparâmetros para TF-IDF(word)+SVD+XGBoost.

    Retorno
    -------
    dict
        Dicionário no formato aceito por GridSearchCV, incluindo:
        - svd__n_components
        - clf__estimator__n_estimators, max_depth, learning_rate
    """
    return {"prep__do_normalize":[True],
            "prep__lowercase":[None],
            "svd__n_components":[128,256],
            "clf__estimator__n_estimators":[200,300],
            "clf__estimator__max_depth":[5,7],
            "clf__estimator__learning_rate":[0.15,0.2]}

# ---------------------------------------------------------------------
# Balanceamento de classes por split:
# - _counts_and_pct: conta e calcula % por classe.
# - report_split_balance: imprime tabelas por (train/val/test), agrega,
#   calcula o desvio-padrão das % entre splits (quanto menor, melhor)
#   e retorna os resumos (por split e conjunto agregado).
# ---------------------------------------------------------------------
def _counts_and_pct(y):
    """Calcula contagens absolutas e percentuais por classe.

    Parâmetros
    ----------
    y : array-like ou pandas.Series
        Sequência de rótulos/classes.

    Retorno
    -------
    pandas.DataFrame
        DataFrame indexado pela classe (ordenado alfabeticamente pelo índice)
        com duas colunas:
            - "count": contagem absoluta por classe.
            - "pct"  : porcentagem (0–100) com duas casas decimais.

    Notas
    -----
    - A porcentagem é calculada como `count / total * 100` e arredondada
        para duas casas decimais.
    """
    vc = pd.Series(y).value_counts().sort_index(); pct = (vc/vc.sum()*100).round(2)
    return pd.DataFrame({"count": vc, "pct": pct})

def report_split_balance(y_train, y_val, y_test):
    """Relata o balanceamento de classes em train/val/test e resume a variância.

    Para cada split, imprime a tabela de contagem e porcentagem por classe.
    Em seguida, agrega os três splits em um único DataFrame e calcula o desvio-padrão
    das porcentagens por classe (quanto menor, melhor o balanceamento entre splits).

    Parâmetros
    ----------
    y_train : array-like
        Rótulos do conjunto de treino.
    y_val : array-like
        Rótulos do conjunto de validação.
    y_test : array-like
        Rótulos do conjunto de teste.

    Retorno
    -------
    rep : Dict[str, pandas.DataFrame]
        Dicionário com as tabelas individuais por split:
        `{"train": df_train, "val": df_val, "test": df_test}`.
        Cada DataFrame possui colunas "count" e "pct".
    joint : pandas.DataFrame
        DataFrame agregado resultante de joins:
            - Colunas: count_train, pct_train, count_val, pct_val, count_test, pct_test
            - Coluna adicional: pct_std (desvio-padrão entre pct_train, pct_val, pct_test)

    Efeitos colaterais
    ------------------
    - Imprime no console os resumos por split e o `describe()` de `pct_std`.

    Observações
    -----------
    - O join é feito a partir do índice de `train` (left join). Em cenários
        estratificados padrão, as mesmas classes costumam aparecer nos três splits.
        Caso alguma classe não apareça em um split, os campos correspondentes
        poderão ficar como NaN no `joint`.
    """
    print("\n=== Balanceamento por split (% por classe) ===")
    rep={}
    for name, arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        dfc = _counts_and_pct(arr); rep[name]=dfc; print(f"\n{name.upper()}:\n{dfc}")
    joint = rep["train"].join(rep["val"], lsuffix="_train", rsuffix="_val") \
                        .join(rep["test"].rename(columns={"count":"count_test","pct":"pct_test"}))
    joint["pct_std"] = joint[["pct_train","pct_val","pct_test"]].std(axis=1)
    print("\nResumo do desvio-padrão das % por classe entre splits (menor = melhor):")
    print(joint["pct_std"].describe())
    return rep, joint

# ---------------------------------------------------------------------
# Anti-overfitting:
# - Detecção de vazamento por duplicatas entre splits (hash de textos).
# - OOF (out-of-fold) para avaliar generalização sem tocar no teste.
# - Learning curve (viés/variância) com CSV + plot.
# - Y-scramble como teste de sanidade (modelo não deve performar bem com rótulos permutados).
# ---------------------------------------------------------------------
def _hash_texts(texts):
    """Gera hashes SHA-1 (hex) para uma coleção de textos.

    Parâmetros
    ----------
    texts : Iterable[str]
        Sequência de strings a serem codificadas como UTF-8 (erros ignorados).

    Retorno
    -------
    List[str]
        Lista de digests hexadecimais SHA-1, um por texto.

    Notas
    -----
    - Útil para deduplicação e criação de grupos estáveis entre splits.
    - SHA-1 aqui é apenas uma função de fingerprint; não use para fins criptográficos.
    """
    return [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts]

def check_duplicates_leakage(X_train, X_val, X_test, plot_dir: Path):
    """Detecta duplicatas internas e entre splits (possível leakage) via hash de texto normalizado.

    Fluxo
    -----
    1) Normaliza cada split com `TextNormalizer` (NFKC + casefold heurístico).
    2) Calcula SHA-1 por amostra e mede:
        - Duplicatas dentro de cada split (chaves repetidas).
        - Interseções entre splits (train∩val, train∩test, val∩test).
    3) Salva `diag_text_leakage.json` com estatísticas.

    Parâmetros
    ----------
    X_train, X_val, X_test : Iterable[str]
        Textos de treino/validação/teste.
    plot_dir : pathlib.Path
        Diretório de plots da execução; o JSON será salvo em `plot_dir.parent`.

    Retorno
    -------
    Dict[str, Any]
        Dicionário com estatísticas por split e interseções.

    Efeitos colaterais
    ------------------
    - Escreve `diag_text_leakage.json`.
    - Imprime mensagens e alerta caso haja interseções > 0.

    Observações
    -----------
    - Caso haja interseções, considere usar split estratificado por grupos (hash).
    """
    norm = TextNormalizer(do_normalize=True, lowercase=None)
    ntr = norm.transform(X_train); nva = norm.transform(X_val); nts = norm.transform(X_test)
    htr, hva, hts = _hash_texts(ntr), _hash_texts(nva), _hash_texts(nts)
    s_tr, s_va, s_ts = set(htr), set(hva), set(hts)
    inter_tr_va = len(s_tr & s_va); inter_tr_ts = len(s_tr & s_ts); inter_va_ts = len(s_va & s_ts)
    def dup_stats(h):
        vc = pd.Series(h).value_counts(); dups = int((vc > 1).sum()); total_dup_rows = int((vc[vc > 1] - 1).sum())
        return {"unique": int(vc.size), "dups_keys": dups, "extra_rows_from_dups": total_dup_rows}
    stats = {"train": dup_stats(htr), "val": dup_stats(hva), "test": dup_stats(hts),
                "cross_split_duplicates": {"train_val": inter_tr_va, "train_test": inter_tr_ts, "val_test": inter_va_ts}}
    out = plot_dir.parent / "diag_text_leakage.json"
    with open(out, "w", encoding="utf-8") as f: json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[anti-of] Checagem de duplicatas/leakage salva em: {out.resolve()}")
    if any(v > 0 for v in stats["cross_split_duplicates"].values()):
        print("[anti-of][ALERTA] Há textos idênticos entre splits. Considere split por grupos (hash).")
    return stats

def oof_diagnostics(pipe, name, X, y, cv_folds, n_jobs, plot_saver):
    """Gera diagnósticos OOF (out-of-fold) para estimar generalização sem tocar no teste.

    Método
    ------
    - Usa `StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)`.
    - Obtém `y_oof` via `cross_val_predict(..., method='predict')`.
    - Calcula macro-F1 e `classification_report`.
    - Salva JSON `artifacts/oof_report_{name}.json`.
    - (Opcional) Plota matriz de confusão OOF.

    Parâmetros
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline (não necessariamente ajustado); será clonado internamente.
    name : str
        Nome do experimento para compor arquivos.
    X : array-like
        Atributos.
    y : array-like
        Rótulos verdadeiros.
    cv_folds : int
        Número de dobras estratificadas.
    n_jobs : int
        Paralelismo para `cross_val_predict`.
    plot_saver : Optional[object]
        Se fornecido, deve expor `.savefig(fig, name, dpi=...)` para salvar o plot.

    Retorno
    -------
    float
        Macro-F1 OOF.

    Efeitos colaterais
    ------------------
    - Salva `oof_report_{name}.json`.
    - Gera e salva `oof_confusion_{name}.png` quando `plot_saver` é fornecido.
    """
    print("\n[anti-of] Gerando OOF (out-of-fold) predictions...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    est = clone(pipe)
    y_oof = cross_val_predict(est, X, y, cv=cv, n_jobs=n_jobs, method="predict", verbose=0)
    macro = f1_score(y, y_oof, average="macro")
    rep = classification_report(y, y_oof, output_dict=True, zero_division=0)
    print(f"[anti-of] OOF macro-F1 = {macro:.4f}")
    report_path = Path("artifacts") / f"oof_report_{name}.json"
    with open(report_path, "w", encoding="utf-8") as f: json.dump({"macro_f1": macro, "report": rep}, f, ensure_ascii=False, indent=2)
    print(f"[anti-of] OOF report salvo em: {report_path.resolve()}")
    if plot_saver is not None:
        classes = np.unique(y); cm = confusion_matrix(y, y_oof, labels=classes)
        fig, ax = plt.subplots(figsize=(9,9))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)
        ax.set_title(f"OOF Confusion - {name}")
        plot_saver.savefig(fig, f"oof_confusion_{name}.png"); plt.tight_layout(); plt.show()
    return macro

def learning_curve_diag(pipe, name, X, y, cv_folds, n_jobs, points, plot_saver):
    """Gera curva de aprendizado (train vs. CV macro-F1) e salva CSV/plot.

    Método
    ------
    - Usa `learning_curve` com `train_sizes = linspace(0.1, 1.0, points)`,
        `scoring='f1_macro'` e `StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)`.
    - Salva `artifacts/learning_curve_{name}.csv` com médias e desvios.

    Parâmetros
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline base (clonado internamente).
    name : str
        Nome do experimento para arquivos.
    X, y : array-like
        Atributos e rótulos.
    cv_folds : int
        Nº de dobras.
    n_jobs : int
        Paralelismo do `learning_curve`.
    points : int
        Nº de pontos (tamanhos de treino) na curva.
    plot_saver : Optional[object]
        Se fornecido, plota e salva `learning_curve_{name}.png`.

    Retorno
    -------
    None

    Efeitos colaterais
    ------------------
    - Salva CSV com colunas: train_size, train_mean, train_std, val_mean, val_std.
    - (Opcional) Salva PNG da curva.
    """
    print("\n[anti-of] Gerando Learning Curve...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    sizes = np.linspace(0.1, 1.0, points)
    est = clone(pipe)
    train_sizes, train_scores, val_scores = learning_curve(
        est, X, y, cv=cv, n_jobs=n_jobs, scoring="f1_macro", train_sizes=sizes, shuffle=True, random_state=SEED
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
    df = pd.DataFrame({"train_size": train_sizes, "train_mean": train_mean, "train_std": train_std,
                        "val_mean": val_mean, "val_std": val_std})
    csv_path = Path("artifacts") / f"learning_curve_{name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[anti-of] Learning curve CSV salvo em: {csv_path.resolve()}")
    if plot_saver is not None:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(train_sizes, train_mean, marker="o", label="Treino (macro-F1)")
        ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
        ax.plot(train_sizes, val_mean, marker="s", label="CV (macro-F1)")
        ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
        ax.set_xlabel("Tamanho de treino"); ax.set_ylabel("Macro-F1"); ax.set_title(f"Learning Curve - {name}")
        ax.legend(loc="lower right")
        plt.tight_layout(); plot_saver.savefig(fig, f"learning_curve_{name}.png"); plt.show()

def y_scramble_diag(pipe, name, X_tr, y_tr, X_val, y_val, n_reps):
    """Executa Y-scramble como teste de sanidade para detectar overfitting/artefatos.

    Método
    ------
    Para `n_reps` repetições:
        1) Permuta `y_tr` com `RandomState(SEED + 1000 + i)`.
        2) Ajusta um clone do `pipe` em `(X_tr, y_perm)`.
        3) Avalia em `(X_val, y_val)` e registra `macro_f1` e `accuracy`.

    Parâmetros
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline base (será clonado a cada repetição).
    name : str
        Nome do experimento (para nomear o CSV).
    X_tr, y_tr : array-like
        Conjunto de treino original.
    X_val, y_val : array-like
        Conjunto de validação para aferição.
    n_reps : int
        Número de repetições de permutação.

    Retorno
    -------
    None

    Efeitos colaterais
    ------------------
    - Salva `artifacts/diag_y_scramble_{name}.csv` com colunas: rep, macro_f1, accuracy.
    - Imprime métricas por repetição e o resumo (média ± desvio) de macro-F1.

    Interpretação
    -------------
    - Se o modelo obtiver desempenho próximo ao acaso com rótulos permutados, é um
        sinal de que não está aprendendo artefatos triviais do pipeline/dados.
    """
    print(f"\n[anti-of] Rodando Y-Scramble com {n_reps} repetições...")
    rows=[]
    for i in range(n_reps):
        rs = np.random.RandomState(SEED + 1000 + i)
        y_perm = rs.permutation(y_tr)
        est = clone(pipe); est.fit(X_tr, y_perm)
        y_pred = est.predict(X_val)
        macro = f1_score(y_val, y_pred, average="macro"); acc = accuracy_score(y_val, y_pred)
        rows.append({"rep": i+1, "macro_f1": float(macro), "accuracy": float(acc)})
        print(f"[anti-of] Y-scramble rep {i+1}: macro-F1={macro:.4f} | acc={acc:.4f}")
    df = pd.DataFrame(rows)
    csv_path = Path("artifacts") / f"diag_y_scramble_{name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[anti-of] Y-scramble CSV salvo em: {csv_path.resolve()}")
    print(f"[anti-of] Y-scramble média macro-F1 = {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")

# ---------------------------------------------------------------------
# Split estratificado por grupos (anti-leak):
# - compute_group_hashes: normaliza texto e gera hash estável (agrupa duplicatas/variações).
# - grouped_stratified_split: faz 80/10/10 estratificado no nível do grupo
#   usando o rótulo majoritário por grupo; cada hash cai em um único split.
#   Retorna máscaras booleanas alinhadas a X/y; checa sobreposição e é reprodutível.
# ---------------------------------------------------------------------
def compute_group_hashes(texts):
    """Gera *group IDs* estáveis a partir de textos normalizados.

    Processo
    --------
    1) Aplica `TextNormalizer` (NFKC + casefold heurístico) em cada texto.
    2) Calcula SHA-1 em hexadecimal do texto normalizado.

    Parâmetros
    ----------
    texts : Iterable[str]
        Coleção de textos (valores não‐string serão convertidos para `str` pelo normalizador).

    Retorno
    -------
    numpy.ndarray
        Array 1D de strings (hex SHA-1, 40 caracteres) com shape `(n_amostras,)`,
        a ser usado como `groups` em splits anti-vazamento.

    Observações
    -----------
    - Textos idênticos (após normalização) terão o mesmo hash, caindo no mesmo grupo.
    - Útil para impedir que duplicatas/variações caiam em splits diferentes.
    """
    norm = TextNormalizer(do_normalize=True, lowercase=None)
    normed = norm.transform(texts)
    return np.array(_hash_texts(normed))

def grouped_stratified_split(X, y, groups, test_size=0.10, val_size=0.10, random_state=SEED):
    """Split 80/10/10 estratificado por classe **no nível de grupo** (anti-leak).

    Ideia
    -----
    Cada grupo (por exemplo, hash do texto normalizado) é alocado *integralmente*
    a **um único** split (train, val ou test), evitando vazamento por duplicatas.
    A estratificação é feita no nível de grupo usando o **rótulo majoritário** de cada grupo.

    Parâmetros
    ----------
    X : array-like
        Atributos (não usados diretamente no split; serve apenas para dimensionar máscaras).
    y : array-like
        Rótulos por amostra (strings ou ints).
    groups : array-like
        IDs de grupo por amostra (ex.: saída de `compute_group_hashes`).
    test_size : float, default=0.10
        Proporção destinada a TEST.
    val_size : float, default=0.10
        Proporção destinada a VAL (do total).
    random_state : int, default=SEED
        Semente para reprodutibilidade do `train_test_split`.

    Retorno
    -------
    m_train, m_val, m_test : Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Máscaras booleanas 1D, alinhadas a `X`/`y`, indicando as amostras de cada split.

    Raises
    ------
    AssertionError
        Se houver sobreposição de grupos entre as máscaras resultantes (não deveria ocorrer).
    ValueError
        Pode ser levantado internamente pelo `train_test_split` caso a estratificação por grupos
        não seja possível (ex.: classes com grupos insuficientes).

    Observações
    -----------
    - O procedimento faz dois `train_test_split` no conjunto de **grupos**:
        1) `train` vs `temp (val+test)` estratificado por rótulo majoritário do grupo.
        2) Dentro de `temp`, separa `val` e `test` preservando a proporção especificada.
    - A proporção final será aproximadamente 80/10/10, sujeita a discretização por grupos.
    """
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({"idx": np.arange(len(X)), "y": y, "g": groups})
    # rótulo majoritário por grupo
    g_counts = df.groupby(["g","y"]).size().reset_index(name="n")
    maj = g_counts.sort_values(["g","n"], ascending=[True,False]).drop_duplicates("g")
    g2label = dict(zip(maj["g"].values, maj["y"].values))
    g_unique = df["g"].unique()
    g_labels = np.array([g2label[g] for g in g_unique])

    # 1) train vs temp (val+test)
    g_train, g_temp, y_train_g, y_temp_g = train_test_split(
        g_unique, g_labels, test_size=(test_size+val_size), random_state=random_state, stratify=g_labels
    )
    # 2) val vs test dentro do temp
    val_prop = val_size/(val_size+test_size)
    g_val, g_test, y_val_g, y_test_g = train_test_split(
        g_temp, y_temp_g, test_size=(1.0 - val_prop), random_state=random_state, stratify=y_temp_g
    )

    def mask_groups(gs):
        s = set(gs); return df["g"].isin(s).values
    m_train, m_val, m_test = mask_groups(g_train), mask_groups(g_val), mask_groups(g_test)

    assert not (m_train & m_val).any() and not (m_train & m_test).any() and not (m_val & m_test).any(), "Sobreposição de grupos entre splits!"
    return m_train, m_val, m_test

# ---------------------------------------------------------------------
# MAIN: orquestra o pipeline de ponta a ponta.
# - Cria diretórios/artefatos e logging (tee para arquivo + console).
# - Patcher do plt.show para salvar plots automaticamente (PlotSaver).
# - Lê flags de ambiente (n_jobs, halving, EDA, diagnósticos, etc.) e configura cache.
# - Carrega dataset, faz EDA opcional e split 80/10/10 com anti-leak por grupos (hash).
# - Executa Grid/HalvingGrid em múltiplos experimentos, salva cv_results e resume validação.
# - Refit do melhor (train+val), avaliação final no TEST (métricas/plots) e diagnósticos anti-overfitting.
# - Persiste modelo, metadados e caminhos dos artefatos; imprime predições de exemplo.
# ---------------------------------------------------------------------
def main():
    """Executa o pipeline completo de LID (Language Identification) de ponta a ponta.

    Visão geral do fluxo
    --------------------
    1) Inicialização de artefatos e logging:
        - Cria `artifacts/` e `artifacts/plots_<timestamp>/`.
        - Redireciona stdout/stderr para um *tee* (console + arquivo `run_<timestamp>.txt`).
        - Patching de `matplotlib.pyplot.show` para salvar automaticamente toda figura gerada
            (classe interna `PlotSaver`).

    2) Configuração de busca/execução:
        - Detecta notebook (para ajustar `n_jobs`).
        - Lê *feature flags* via variáveis de ambiente (HalvingGrid, cache do Pipeline, EDA, etc.).
        - Configura cache opcional do `sklearn.Pipeline` via `joblib.Memory` (define global `MEM`).

    3) Dados e EDA (opcional):
        - Carrega dataset (`load_dataset`) e infere colunas (`smart_column_guess`).
        - Limpa linhas vazias, corrige rótulos canônicos (“Portugese”→“Portuguese”).
        - Plota distribuição de classes, histogramas/boxplots de comprimento e *char n-grams* (se `SKIP_EDA=0`).

    4) Split anti-vazamento:
        - Se `SPLIT_BY_GROUPS=1` (default), cria *groups* por hash de texto normalizado e faz split 80/10/10
            estratificado no nível de grupo (`grouped_stratified_split`).
        - Caso contrário, usa `train_test_split` estratificado padrão.

    5) Seleção de modelo:
        - Define experimentos (TF-IDF char + LinearSVC; TF-IDF word + LogReg; e opcionais).
        - Para cada experimento: `GridSearchCV` (ou `HalvingGridSearchCV` se habilitado), salva `cv_results_*.csv`,
            imprime métricas em *val* e mantém o melhor por Macro-F1.

    6) Refit e avaliação final:
       - Reajusta o melhor *pipeline* em `train+val` e avalia no *TEST* (`evaluate_on_split`),
         gerando relatório JSON, matrizes de confusão (PNG), Top-3 accuracy, métricas por *script* e CSV de erros.

    7) Diagnósticos anti-overfitting (opcional):
        - Checagem de duplicatas entre splits (`diag_text_leakage.json`).
        - OOF (macro-F1 + confusão OOF), curva de aprendizado (CSV+PNG) e Y-scramble (CSV).

    8) Persistência e metadados:
        - Salva o melhor modelo (`best_langid_<exp>.joblib`).
        - Gera `metadata_ext.json` com resumo da execução, caminhos de artefatos e classes.

    9) *Smoke test* de inferência:
        - Imprime predições do melhor modelo em 5 frases exemplo.

    Variáveis de ambiente suportadas
    --------------------------------
    - `GS_N_JOBS`           : nº de processos no Grid/Halving (default = CPUs fora de notebook, 1 em notebook).
    - `USE_HALVING`         : "1" para usar `HalvingGridSearchCV` (se disponível).
    - `SKL_CACHE`           : "1" ativa cache do Pipeline; "0" desativa (Windows desativa por padrão).
    - `SKIP_EDA`            : "1" para pular EDA (plots/prints).
    - `CV_FOLDS`            : dobras na validação cruzada (default=3).
    - `SEARCH_SUBSAMPLE`    : N para amostrar N exemplos em GridSearch (acelera).
    - `CALIB_AT_END`        : "1" calibra `LinearSVC` (sigmoid, cv=3) após refit final.
    - `RUN_HEAVY`           : "1" inclui modelos extras (NB, W2V/D2V, XGB).
    - `ANTI_OVERFITTING`    : "1" executa diagnósticos anti-OF.
    - `Y_SCRAMBLE_N`        : repetições do Y-scramble (default=5).
    - `OOF_USE_TRVAL`       : "1" usa `train+val` nos diagnósticos OOF (senão só `train`).
    - `LC_POINTS`           : nº de pontos da curva de aprendizado (default=6).
    - `SPLIT_BY_GROUPS`     : "1" ativa split por hash de texto (anti-leak) [default].
    - (fora deste escopo, mas relevantes aos experimentos): `XGB_USE_GPU`, `W2V_WORKERS`.

    Arquivos e diretórios gerados (em `artifacts/`)
    -----------------------------------------------
    - `run_<timestamp>.txt`                     : log completo da execução.
    - `plots_<timestamp>/...png`                : todos os gráficos salvos automaticamente.
    - `cv_results_<exp>.csv`                    : resultados de Grid/Halving por experimento.
    - `validation_results.csv`                  : ranking em validação.
    - `best_langid_<exp>.joblib`               : modelo final.
    - `classification_report_test_<exp>.json`   : relatório do TEST.
    - `errors_test_<exp>.csv`                   : exemplos de erros (se houver).
    - `diag_text_leakage.json`                  : duplicatas/vazamento entre splits.
    - `oof_report_<exp>.json`                   : métricas OOF.
    - `learning_curve_<exp>.csv` (+ PNG)        : curva de aprendizado.
    - `diag_y_scramble_<exp>.csv`               : resultados do Y-scramble.
    - `metadata_ext.json`                       : metadados consolidados da execução.

    Dependências internas
    ---------------------
    Requer as funções/classes definidas no módulo: `_running_in_notebook`, `load_dataset`,
    `smart_column_guess`, `plot_class_distribution`, `plot_length_distributions`,
    `top_char_ngrams_per_language`, `compute_group_hashes`, `grouped_stratified_split`,
    `report_split_balance`, `make_*`/`grid_*` de modelos, `evaluate_on_split`,
    `check_duplicates_leakage`, `oof_diagnostics`, `learning_curve_diag`, `y_scramble_diag`.

    Raises
    ------
    RuntimeError
        Quando nenhum experimento termina com sucesso (p.ex., dependências ausentes ou dados inválidos).

    Retorno
    -------
    None
        Função *script entrypoint*; efeitos são a geração de artefatos e prints no console.

    Exemplo
    -------
    >>> if __name__ == "__main__":
    ...     main()
    """
    global MEM, ARTIFACTS_DIR
    RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    ARTIFACTS_DIR = Path("artifacts"); ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = ARTIFACTS_DIR / f"plots_{RUN_STAMP}"; PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    class _Tee(io.TextIOBase):
        """Redireciona a escrita para múltiplos *streams* (console + arquivo)."""
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                try: st.write(s); st.flush()
                except Exception: pass
            return len(s)
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass

    _LOG_PATH = ARTIFACTS_DIR / f"run_{RUN_STAMP}.txt"
    _log_file = open(_LOG_PATH, "w", encoding="utf-8")
    _ORIG_STDOUT = sys.__stdout__; _ORIG_STDERR = sys.__stderr__
    sys.stdout = _Tee(sys.stdout, _log_file); sys.stderr = sys.stdout

    @atexit.register
    def _close_log_file():
        """Restaura stdout/stderr e fecha o arquivo de log no término do processo."""
        try: _log_file.flush(); _log_file.close()
        except Exception: pass
        try:
            sys.stdout = _ORIG_STDOUT; sys.stderr = _ORIG_STDERR
            sys.__stdout__.write(f"[i] Log salvo em: {_LOG_PATH.resolve()}\n"); sys.__stdout__.flush()
        except Exception: pass

    print(f"[i] Logging ativo. Output -> {_LOG_PATH.resolve()}")
    print(f"[i] Plots -> {PLOTS_DIR.resolve()}")

    class PlotSaver:
        """Helper para salvar figuras e *auto-nomear* PNGs quando `plt.show()` é chamado."""
        def __init__(self, outdir):
            self.dir = Path(outdir); self.dir.mkdir(parents=True, exist_ok=True)
            self.counter = 0
        def savefig(self, fig=None, name=None, dpi=150):
            if fig is None: fig = plt.gcf()
            if name is None: self.counter += 1; name = f"plot_{self.counter:03d}.png"
            path = self.dir / name
            setattr(fig, "_saved_named", True); setattr(fig, "_saved_path", str(path))
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"[plot] Figura salva: {path.resolve()}"); return path

    plot_saver = PlotSaver(PLOTS_DIR); _ORIG_SHOW = plt.show
    @functools.wraps(_ORIG_SHOW)
    def _patched_show(*args, **kwargs):
        """Intercepta `plt.show()` para garantir que toda figura seja persistida em PNG."""
        fig = plt.gcf()
        if not getattr(fig, "_saved_named", False):
            plot_saver.counter += 1; name = f"plot_{plot_saver.counter:03d}.png"
            path = plot_saver.dir / name
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[plot:auto] Figura salva: {path.resolve()}")
        return _ORIG_SHOW(*args, **kwargs)
    try: _patched_show.__signature__ = inspect.signature(_ORIG_SHOW)
    except Exception: pass
    plt.show = _patched_show

    IN_NOTEBOOK = _running_in_notebook()
    _default_jobs = (1 if IN_NOTEBOOK else multiprocessing.cpu_count())
    GS_N_JOBS = int(os.getenv("GS_N_JOBS", str(_default_jobs)))
    print(f"[i] GridSearch backend = 'loky', n_jobs = {GS_N_JOBS} (IN_NOTEBOOK={IN_NOTEBOOK})")

    USE_HALVING = os.getenv("USE_HALVING", "0") == "1"
    if USE_HALVING:
        try:
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingGridSearchCV
            HAVE_HALVING = True
        except Exception:
            HAVE_HALVING = False
            print("[Aviso] USE_HALVING=1, mas HalvingGridSearchCV indisponível. Usando GridSearchCV comum.")
    else:
        HAVE_HALVING = False

    SKL_CACHE_DIR = ARTIFACTS_DIR / "skl_cache"
    def _make_memory():
        """Decide e cria `joblib.Memory` para o Pipeline conforme SO/variável de ambiente."""
        env = os.getenv("SKL_CACHE")
        if env == "0":
            print("[i] SKL_CACHE=0 → cache do Pipeline desativado por ambiente."); return None
        if os.name == "nt" and env != "1":
            print("[i] Windows detectado → desativando cache do Pipeline (evita PicklingError)."); return None
        return Memory(location=str(SKL_CACHE_DIR), verbose=0)
    global MEM; MEM = _make_memory()

    SKIP_EDA = os.getenv("SKIP_EDA", "0") == "1"
    CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))
    SEARCH_SUBSAMPLE = int(os.getenv("SEARCH_SUBSAMPLE", "0"))
    CALIB_AT_END = os.getenv("CALIB_AT_END", "0") == "1"
    RUN_HEAVY = os.getenv("RUN_HEAVY", "1") == "1"  # default roda tudo
    ANTI_OVERFITTING = os.getenv("ANTI_OVERFITTING", "1") == "1"
    Y_SCRAMBLE_N = int(os.getenv("Y_SCRAMBLE_N", "5"))
    OOF_USE_TRVAL = os.getenv("OOF_USE_TRVAL", "1") == "1"
    LC_POINTS = int(os.getenv("LC_POINTS", "6"))
    SPLIT_BY_GROUPS = os.getenv("SPLIT_BY_GROUPS", "1") == "1"  # <<< NOVO: default ON

    # 1) Dados
    df_raw = load_dataset()
    print("Amostras e colunas:", df_raw.shape, list(df_raw.columns))
    TEXT_COL, LABEL_COL = smart_column_guess(df_raw)
    print(f"Coluna de TEXTO: {TEXT_COL!r} | Coluna de RÓTULO: {LABEL_COL!r}")

    df = df_raw[[TEXT_COL, LABEL_COL]].dropna().copy()
    df = df[(df[TEXT_COL].astype(str).str.strip()!="") & (df[LABEL_COL].astype(str).str.strip()!="")]
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    LABEL_CANON = {"Portugese": "Portuguese"}
    before = df[LABEL_COL].value_counts().get("Portugese", 0)
    if before: print(f"[i] Canonizando rótulos: 'Portugese' -> 'Portuguese' (ocorrências: {before})")
    df[LABEL_COL] = df[LABEL_COL].replace(LABEL_CANON)

    # 2) EDA
    if not SKIP_EDA:
        print("\n===== EDA: Distribuição de classes =====")
        cls_counts = df[LABEL_COL].value_counts(); print(cls_counts); plot_class_distribution(cls_counts, plot_saver)
        print("\n===== EDA: Comprimentos ====="); plot_length_distributions(df, TEXT_COL, LABEL_COL, plot_saver)
        print("\n===== EDA: Top char n-grams ====="); top_char_ngrams_per_language(df, TEXT_COL, LABEL_COL, n=3, topk=10)

    # 3) Split 80/10/10 — por grupos (hash) se habilitado
    X_all = df[TEXT_COL].astype(str).values
    y_all = df[LABEL_COL].values

    if SPLIT_BY_GROUPS:
        print("[i] SPLIT_BY_GROUPS=1 → criando grupos por hash do texto normalizado (sem vazamento).")
        groups = compute_group_hashes(X_all)
        m_train, m_val, m_test = grouped_stratified_split(X_all, y_all, groups, test_size=0.10, val_size=0.10, random_state=SEED)
        X_train, y_train = X_all[m_train], y_all[m_train]
        X_val,   y_val   = X_all[m_val],   y_all[m_val]
        X_test,  y_test  = X_all[m_test],  y_all[m_test]
    else:
        print("[i] SPLIT_BY_GROUPS=0 → usando train_test_split estratificado tradicional.")
        X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.10, random_state=SEED, stratify=y_all)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=SEED, stratify=y_temp)

    print("\nSplit:", len(X_train), "treino |", len(X_val), "val |", len(X_test), "teste")
    if not SKIP_EDA: report_split_balance(y_train, y_val, y_test)

    # 4) CV + seleção
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    exp_results=[]; best_global={"name": None, "score": -1, "pipe": None, "params": None}

    def _make_search(estimator, param_grid, scoring, cv, n_jobs=GS_N_JOBS, verbose=1):
        """Cria GridSearchCV ou HalvingGridSearchCV (se habilitado e disponível)."""
        if HAVE_HALVING:
            from sklearn.model_selection import HalvingGridSearchCV
            return HalvingGridSearchCV(estimator=estimator, param_grid=param_grid, factor=3,
                                        scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)
        else:
            return GridSearchCV(estimator=estimator, param_grid=param_grid,
                                scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)

    X_train_search, y_train_search = X_train, y_train
    if SEARCH_SUBSAMPLE and len(X_train) > SEARCH_SUBSAMPLE:
        rs = np.random.RandomState(SEED)
        idx = rs.choice(len(X_train), size=SEARCH_SUBSAMPLE, replace=False)
        X_train_search = X_train[idx]; y_train_search = y_train[idx]
        print(f"[i] SEARCH_SUBSAMPLE ativo: usando {len(X_train_search)} exemplos (de {len(X_train)}) no GridSearch.")

    EXPERIMENTS_TO_RUN = [
        ("tfidf_char_linsvm", make_tfidf_char_linsvm, grid_tfidf_char_linsvm),
        ("tfidf_word_logreg", make_tfidf_word_logreg, grid_tfidf_word_logreg),
    ]
    if RUN_HEAVY:
        EXPERIMENTS_TO_RUN += [("count_char_nb", make_count_char_nb, grid_count_char_nb)]
        if HAVE_GENSIM:
            EXPERIMENTS_TO_RUN += [("w2v_logreg", make_w2v_logreg, grid_w2v_logreg),
                                    ("d2v_logreg", make_d2v_logreg, grid_d2v_logreg)]
        if HAVE_XGB:
            EXPERIMENTS_TO_RUN += [("xgb_tfidf_word", make_xgb_tfidf_word, grid_xgb_tfidf_word)]

    EXP_REGISTRY = {name: (make, grid) for name, make, grid in EXPERIMENTS_TO_RUN}

    for exp_name, make_fn, grid_fn in EXPERIMENTS_TO_RUN:
        pipe = make_fn()
        if pipe is None:
            print(f"[Aviso] Pulando {exp_name} (dependência ausente)."); continue
        param_grid = grid_fn()
        print(f"\n== Treinando/ajustando: {exp_name} ==")
        print("Grid:", {k: v if isinstance(v, list) else [v] for k,v in param_grid.items()})
        start = time.time()
        try:
            gs = _make_search(estimator=pipe, param_grid=param_grid,
                                scoring="f1_macro", cv=cv, n_jobs=GS_N_JOBS, verbose=1)
            gs.fit(X_train_search, y_train_search)
        except Exception as e:
            print(f"[ERRO] {exp_name} falhou e será ignorado: {repr(e)}"); continue
        elapsed = time.time() - start

        # cv_results_
        try:
            cv_path = ARTIFACTS_DIR / f"cv_results_{exp_name}.csv"
            pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False, encoding="utf-8")
            print(f"[i] CV results salvos em: {cv_path.resolve()}")
        except Exception as e:
            print(f"[Aviso] Falha ao salvar cv_results_ de {exp_name}: {e}")

        val_pred = gs.best_estimator_.predict(X_val)
        acc = accuracy_score(y_val, val_pred); f1m = f1_score(y_val, val_pred, average="macro"); f1mi = f1_score(y_val, val_pred, average="micro")
        print(f"[{exp_name}] val_acc={acc:.4f} | val_macroF1={f1m:.4f} | params={gs.best_params_} | tempo={elapsed/60:.1f} min")
        exp_results.append({"exp": exp_name, "best_params": gs.best_params_, "val_acc": acc,
                            "val_f1_macro": f1m, "val_f1_micro": f1mi, "train_time_sec": elapsed})
        if f1m > best_global["score"]:
            best_global = {"name": exp_name, "score": f1m, "pipe": gs.best_estimator_, "params": gs.best_params_}

    # 5) Resumo + gráfico
    res_df = pd.DataFrame(exp_results).sort_values("val_f1_macro", ascending=False)
    print("\nResumo validação (ordenado por Macro-F1):"); print(res_df)
    if len(res_df):
        res_csv = ARTIFACTS_DIR / "validation_results.csv"; res_df.to_csv(res_csv, index=False, encoding="utf-8")
        print(f"[i] Resultados de validação salvos em: {res_csv.resolve()}")
        if not SKIP_EDA:
            figr, axr = plt.subplots(figsize=(8, max(3, 0.5*len(res_df))))
            axr.barh(res_df["exp"], res_df["val_f1_macro"])
            axr.set_xlabel("Val Macro-F1"); axr.set_title("Val Macro-F1 por experimento")
            for i,(m,v) in enumerate(zip(res_df["exp"], res_df["val_f1_macro"])): axr.text(v+0.001, i, f"{v:.3f}", va="center")
            plt.tight_layout(); plot_saver.savefig(figr, "results_val_macroF1_by_experiment.png"); plt.show()

    # 6) Guard + refit(train+val) + teste
    if not len(res_df) or best_global["pipe"] is None:
        raise RuntimeError("Nenhum experimento executou com sucesso. Verifique dependências/dados.")
    print(f"\n>> Melhor experimento (val): {best_global['name']} | val_macroF1={best_global['score']:.4f}")
    X_trval = np.concatenate([X_train, X_val]); y_trval = np.concatenate([y_train, y_val])

    make_fn_best, grid_fn_best = EXP_REGISTRY[best_global["name"]]
    final_pipe = make_fn_best(); final_pipe.set_params(**best_global["params"]); final_pipe.fit(X_trval, y_trval)

    if CALIB_AT_END and isinstance(final_pipe.named_steps["clf"], LinearSVC):
        print("[i] CALIB_AT_END=1 → calibrando LinearSVC no final (sigmoid, cv=3).")
        steps = list(final_pipe.named_steps.items()); feats = Pipeline(steps[:-1], memory=MEM); clf = steps[-1][1]
        X_trval_feats = feats.transform(X_trval)
        calib = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3); calib.fit(X_trval_feats, y_trval)
        from sklearn.pipeline import make_pipeline
        final_pipe = make_pipeline(*(s[1] for s in steps[:-1]), calib)

    print("\n>> Avaliação FINAL no TEST com refit(train+val):")
    test_metrics = evaluate_on_split(final_pipe, X_test, y_test, title="TEST", exp_name=best_global["name"],
                                        save_prefix=None, plot_saver=(None if os.getenv("SKIP_EDA","0")=="1" else plot_saver))

    # 7) Anti-overfitting
    if ANTI_OVERFITTING:
        try:
            check_duplicates_leakage(X_train, X_val, X_test, PLOTS_DIR)
            X_oof = np.concatenate([X_train, X_val]) if OOF_USE_TRVAL else X_train
            y_oof = np.concatenate([y_train, y_val]) if OOF_USE_TRVAL else y_train
            oof_diagnostics(final_pipe, best_global["name"], X_oof, y_oof, CV_FOLDS, GS_N_JOBS,
                            None if SKIP_EDA else plot_saver)
            learning_curve_diag(final_pipe, best_global["name"], X_oof, y_oof, CV_FOLDS, GS_N_JOBS,
                                int(os.getenv("LC_POINTS", "6")), None if SKIP_EDA else plot_saver)
            y_scramble_diag(final_pipe, best_global["name"], X_train, y_train, X_val, y_val, int(os.getenv("Y_SCRAMBLE_N", "5")))
        except Exception as e:
            print(f"[anti-of][Aviso] Diagnósticos falharam: {e}")

    # 8) Artefatos / metadados
    try:
        best_model_path = ARTIFACTS_DIR / f"best_langid_{best_global['name']}.joblib"
        joblib.dump(final_pipe, best_model_path)
        print(f"\nModelo salvo em: {best_model_path.resolve()}")
    except Exception as e:
        print(f"[Aviso] Falha ao salvar modelo: {e}")

    try:
        split_reports, _ = report_split_balance(y_train, y_val, y_test)
        class_dist = {
            "train": split_reports["train"]["count"].to_dict(),
            "val": split_reports["val"]["count"].to_dict(),
            "test": split_reports["test"]["count"].to_dict()
        }
    except Exception:
        class_dist = {}

    label_to_index = {lbl: i for i, lbl in enumerate(final_pipe.named_steps["clf"].classes_)}
    meta = {
        "run_stamp": RUN_STAMP,
        "best_experiment": best_global["name"],
        "best_params": best_global["params"],
        "validation_results": res_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "seed": SEED,
        "notes": {
            "split_by_groups": bool(SPLIT_BY_GROUPS),
            "normalization": "prep__do_normalize=True; lowercase=None/True (word TF-IDF)",
            "vectorizers": "TF-IDF (char/word), Count (char), W2V/D2V",
            "xgb": "TF-IDF word + SVD + XGB (gpu_hist opcional) via LabelEncodedClassifier",
            "script_groups": sorted(set(LANG_TO_SCRIPT.values()))
        },
        "splits": {"sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                    "class_distribution": class_dist},
        "classes": list(final_pipe.named_steps["clf"].classes_),
        "label_to_index": label_to_index,
        "artifacts": {
            "log_path": str(_LOG_PATH),
            "plots_dir": str(PLOTS_DIR),
            "model_path": str(ARTIFACTS_DIR / f"best_langid_{best_global['name']}.joblib"),
            "metadata_path": str(ARTIFACTS_DIR / "metadata_ext.json"),
            "validation_csv": str(ARTIFACTS_DIR / "validation_results.csv"),
            "classification_report_json": test_metrics.get("classification_report_path"),
            "errors_csv": test_metrics.get("errors_csv_path"),
            "dup_leakage_json": str(ARTIFACTS_DIR / "diag_text_leakage.json"),
            "oof_report": str(ARTIFACTS_DIR / f"oof_report_{best_global['name']}.json"),
            "learning_curve_csv": str(ARTIFACTS_DIR / f"learning_curve_{best_global['name']}.csv"),
            "y_scramble_csv": str(ARTIFACTS_DIR / f"diag_y_scramble_{best_global['name']}.csv"),
        },
        "search": {"use_halving": bool(HAVE_HALVING),
                    "cv_folds": CV_FOLDS,
                    "gridsearch_n_jobs": GS_N_JOBS,
                    "gridsearch_backend": "loky"}
    }
    with open(ARTIFACTS_DIR / "metadata_ext.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Metadados salvos em: {(ARTIFACTS_DIR / 'metadata_ext.json').resolve()}")

    # 9) Inferência de exemplo
    examples = [
        "A vida é bela e a modelagem de dados é fascinante.",
        "The quick brown fox jumps over the lazy dog.",
        "هذا مثال لجملة قصيرة باللغة العربية.",
        "これは日本語の短い文章の例です。",
        "Проверка классификатора на русском языке."
    ]
    preds = final_pipe.predict(examples)
    print("\nPredições (melhor experimento refit) em frases de exemplo:")
    for txt, p in zip(examples, preds):
        print(f"- ({p}) {txt}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Notas:
# - O script roda em modo standalone, mas pode ser adaptado para notebooks.
# - Requer bibliotecas específicas instaladas (ver dependências no início).
# - Configurações via variáveis de ambiente permitem customização sem alterar o código.
# - O uso de cache (MEM) melhora a performance em pipelines complexos.
# - Anti-overfitting inclui diagnósticos, OOF, learning curves e Y-scramble.
# - Split por grupos evita vazamento de dados entre splits, garantindo reprodutibilidade.
# - Artefatos (modelos, logs, plots) são salvos em diretórios organizados.
# - Plots são salvos automaticamente com nomes únicos para fácil identificação.
# - O script é robusto a erros comuns e fornece mensagens claras sobre o progresso.

# - **Owner:** Guilherme de Oliveira Silva
# - **Last updated:** 2025-08-24 (America/São_Paulo)
# Fim do script.