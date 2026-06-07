"""
classifier.py - Document clustering using TF-IDF + KMeans for LDRM.
Assigns each document a cluster label and updates the database accordingly.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

from nlp import preprocess_text, build_tfidf_matrix
import db

# Default number of clusters
DEFAULT_K = 3


def cluster_documents(k: int = DEFAULT_K) -> dict[int, str]:
    """
    Cluster all documents in the database using TF-IDF + KMeans.

    Steps:
      1. Load all document contents from the DB.
      2. Preprocess text with jieba.
      3. Vectorise with TF-IDF.
      4. Run KMeans with k clusters.
      5. Update each document's category in the DB.

    Returns a dict mapping {doc_id: category_label}.
    If fewer than k documents exist, k is clamped to the number of docs.
    Returns an empty dict if there are no documents.
    """
    rows = db.get_all_contents()  # list of (id, content)
    if not rows:
        return {}

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    preprocessed = [preprocess_text(t) for t in texts]

    vectorizer, matrix = build_tfidf_matrix(preprocessed)

    if vectorizer is None:
        # Fallback: assign all documents to a single cluster
        for doc_id in ids:
            db.update_document_category(doc_id, "簇 1")
        return {doc_id: "簇 1" for doc_id in ids}

    # Normalise rows to unit length (cosine similarity approximation)
    matrix_norm = normalize(matrix, norm="l2")

    # Clamp k to the number of available documents
    actual_k = min(k, len(ids))
    if actual_k < 2:
        label = "簇 1"
        db.update_document_category(ids[0], label)
        return {ids[0]: label}

    kmeans = KMeans(
        n_clusters=actual_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    labels = kmeans.fit_predict(matrix_norm)

    result = {}
    for doc_id, cluster_idx in zip(ids, labels):
        category = f"簇 {cluster_idx + 1}"
        db.update_document_category(doc_id, category)
        result[doc_id] = category

    return result


def get_cluster_summary(k: int = DEFAULT_K) -> dict[str, list[str]]:
    """
    Run clustering and return a summary dict mapping cluster names to
    a list of document titles in that cluster.
    Useful for display purposes.
    """
    cluster_map = cluster_documents(k)  # {doc_id: category}
    docs = db.get_all_documents()

    summary: dict[str, list[str]] = {}
    id_to_title = {doc["id"]: doc["title"] for doc in docs}

    for doc_id, category in cluster_map.items():
        summary.setdefault(category, []).append(
            id_to_title.get(doc_id, f"Doc {doc_id}")
        )

    return summary
