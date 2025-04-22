from collections import Counter
import pandas as pd
from math import log

# Load article text from file
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Compute raw frequency (needed for TF part of TF-IDF)
def compute_raw_frequency(tokens, vocab):
    count = Counter(tokens)
    return {term: count.get(term, 0) for term in vocab}

# Compute IDF
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))  # log(N / df(t))
    return idf_dict

# Compute TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    return {term: tf_vector[term] * idf[term] for term in vocab}

# Article titles (matching saved .txt files)
titles = [
    "Robotics",
    "Aerospace",
    "Archaeology",
    "Cryptography",
    "DNA_replication"
]

# Load and tokenize documents
documents = [load_article_from_file(title) for title in titles]
tokenized_docs = [doc.lower().split() for doc in documents]

# Create vocabulary
vocab = set(word for doc in tokenized_docs for word in doc)
vocab_list = list(vocab)

# Compute IDF once
idf = compute_idf(tokenized_docs, vocab)

# Compute TF-IDF for each document
tfidf_vectors = []
for doc in tokenized_docs:
    tf_vector = compute_raw_frequency(doc, vocab)
    tfidf_vector = compute_tfidf(tf_vector, idf, vocab)
    tfidf_vectors.append(tfidf_vector)

# Create TF-IDF matrix
tfidf_matrix = pd.DataFrame(tfidf_vectors, columns=vocab_list).fillna(0)

# Show the TF-IDF matrix
print("Term-Document Matrix (TF-IDF Weights):")
print(tfidf_matrix)
