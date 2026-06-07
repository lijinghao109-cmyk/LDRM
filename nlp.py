"""
nlp.py - Natural Language Processing utilities for LDRM.
Provides text preprocessing (jieba Chinese segmentation) and
TF-IDF based keyword extraction.
"""

import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

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


def get_term_frequencies(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """
    Return the top-N most frequent terms in the text after preprocessing.
    Uses the same tokenization and stopword filtering as keyword extraction.
    """
    preprocessed = preprocess_text(text)
    if not preprocessed.strip():
        return []

    tokens = preprocessed.split()
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1

    sorted_terms = sorted(freq.items(), key=lambda pair: (-pair[1], pair[0]))
    return sorted_terms[:top_n]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Perform a simple extractive summary by scoring sentences using term frequency.
    Returns the highest-scoring sentences in their original order.
    """
    if not text or not text.strip():
        return ""

    # Split by sentence delimiters (Chinese punctuation or newlines)
    raw_sentences = re.split(r'[。！？\n]+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return "\n".join(sentences)

    freq_map = {term: count for term, count in get_term_frequencies(text, top_n=200)}
    scores = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = preprocess_text(sentence).split()
        score = sum(freq_map.get(token, 0) for token in sentence_tokens)
        if len(sentence_tokens) == 0:
            score = 0
        scores.append((score, idx, sentence))

    scores.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scores[:max_sentences], key=lambda item: item[1])
    return "\n".join([item[2] for item in selected])


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


def calculate_doc_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two documents using TF-IDF.
    Returns a value between 0 and 1 (1 = identical, 0 = completely different).
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    preprocessed1 = preprocess_text(text1)
    preprocessed2 = preprocess_text(text2)

    if not preprocessed1 or not preprocessed2:
        return 0.0

    vectorizer, matrix = build_tfidf_matrix([preprocessed1, preprocessed2])
    if vectorizer is None or matrix.shape[0] < 2:
        return 0.0

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(similarity)


def extract_topics_lda(texts: list[str], num_topics: int = 3, passes: int = 10) -> dict:
    """
    Extract topics from a corpus using Latent Dirichlet Allocation (LDA).
    Returns a dict with:
      - 'topics': list of topic descriptions
      - 'doc_topics': list of (doc_id, topic_distribution) tuples
    If corpus is too small, returns empty dict.
    """
    if not texts or len(texts) < 2:
        return {}

    preprocessed = [preprocess_text(t) for t in texts]
    preprocessed = [t for t in preprocessed if t.strip()]

    if len(preprocessed) < 2:
        return {}

    num_topics = min(num_topics, len(preprocessed))

    try:
        # Use CountVectorizer for LDA (LDA works better with term frequencies)
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\S+",
        )
        term_matrix = vectorizer.fit_transform(preprocessed)

        if term_matrix.shape[0] < 2:
            return {}

        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=passes,
            learning_method="online",
            n_jobs=-1,
        )
        lda.fit(term_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx in range(num_topics):
            top_indices = lda.components_[topic_idx].argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_str = "、".join(top_words)
            topics.append(f"主题 {topic_idx+1}: {topic_str}")

        doc_topic_dist = lda.transform(term_matrix)
        doc_topics = [(i, dist) for i, dist in enumerate(doc_topic_dist)]

        return {
            "topics": topics,
            "doc_topics": doc_topics,
        }
    except Exception:
        return {}
