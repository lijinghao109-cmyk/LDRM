"""
nlp.py - Natural Language Processing utilities for LDRM.
Provides text preprocessing (jieba Chinese segmentation) and
TF-IDF based keyword extraction.
"""

import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

# Basic Chinese and English stopwords to filter out noise
STOPWORDS = {
    # Chinese
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "里", "来",
    "对", "我们", "他", "她", "它", "们", "这个", "那个", "什么",
    "如果", "但是", "因为", "所以", "而且", "还是", "或者", "虽然",
    "然后", "可以", "已经", "通过", "进行", "使用", "一些", "以及",
    "其他", "这些", "那些", "时候", "方式", "方面", "问题", "情况",
    "工作", "发展", "研究", "分析", "结果", "可能", "需要", "同时",
    "根据", "表示", "目前", "相关", "主要", "认为", "提出", "提供",
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "we", "our", "you", "your", "he", "she", "his", "her",
    "as", "if", "so", "not", "no", "can", "about", "up", "out",
    "also", "than", "then", "more", "when", "which", "who", "how",
    "all", "both", "each", "few", "into", "through", "during",
    "such", "other", "only", "same", "very", "just", "because",
}


def preprocess_text(text: str) -> str:
    """
    Tokenize and clean a text string:
    - Use jieba for Chinese word segmentation
    - Remove punctuation and numbers
    - Filter stopwords and single-character tokens (for Chinese noise reduction)
    Returns a space-joined string of tokens suitable for scikit-learn vectorizers.
    """
    if not text or not text.strip():
        return ""

    # Remove HTML-like tags and URLs if any
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)

    # Jieba segmentation (handles both Chinese and mixed content)
    tokens = jieba.cut(text, cut_all=False)

    cleaned = []
    for token in tokens:
        token = token.strip()
        # Remove punctuation and whitespace-only tokens
        if re.fullmatch(r"[\s\W\d]+", token):
            continue
        # Skip stopwords and very short Chinese tokens (single char is often noise)
        if token.lower() in STOPWORDS:
            continue
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            # single Chinese character – likely noise, skip
            continue
        cleaned.append(token)

    return " ".join(cleaned)


def build_tfidf_matrix(corpus: list[str]):
    """
    Build a TF-IDF matrix from a list of preprocessed text strings.
    Returns (vectorizer, matrix).
    If the corpus is empty or all documents are blank, returns (None, None).
    """
    non_empty = [doc for doc in corpus if doc.strip()]
    if not non_empty:
        return None, None

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,        # allow terms appearing in a single doc (small corpora)
        max_df=0.95,
        token_pattern=r"(?u)\S+",  # any non-whitespace sequence
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix
    except ValueError:
        return None, None


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract the top-N keywords from a single document using TF-IDF.
    Returns a list of keyword strings.
    """
    preprocessed = preprocess_text(text)
    if not preprocessed.strip():
        return []

    vectorizer, matrix = build_tfidf_matrix([preprocessed])
    if vectorizer is None:
        return []

    # matrix shape is (1, vocab_size); get the single row as array
    scores = matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    # Pair scores with terms and sort descending
    scored = sorted(zip(scores, feature_names), reverse=True)
    keywords = [term for score, term in scored if score > 0]
    return keywords[:top_n]


def extract_keywords_from_corpus(
    texts: list[str], top_n: int = 10
) -> list[list[str]]:
    """
    Extract top-N keywords for each document in a corpus.
    Uses the global TF-IDF vocabulary so term importance is relative to the
    whole collection (better than per-document TF-IDF for ranking).
    Returns a list of keyword lists, one per document.
    """
    preprocessed = [preprocess_text(t) for t in texts]
    vectorizer, matrix = build_tfidf_matrix(preprocessed)

    if vectorizer is None:
        return [[] for _ in texts]

    feature_names = vectorizer.get_feature_names_out()
    result = []
    dense = matrix.toarray()

    for row in dense:
        scored = sorted(zip(row, feature_names), reverse=True)
        keywords = [term for score, term in scored if score > 0]
        result.append(keywords[:top_n])

    return result
