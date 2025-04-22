# import os
import pandas as pd
from collections import Counter
# from pathlib import Path

# Function to load Wikipedia article text from a local file
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Function to compute raw frequency (without normalization)
def compute_raw_frequency(tokens, vocab):
    count = Counter(tokens)
    return {term: count.get(term, 0) for term in vocab}

# List of Wikipedia article titles (same as filenames without .txt)
titles = [
    "Robotics",
    "Aerospace",
    "Archaeology",
    "Cryptography",
    "DNA_replication"
]

# Load articles from local text files
documents = [load_article_from_file(title) for title in titles]

# Tokenize and lowercase
tokenized_docs = [doc.lower().split() for doc in documents]

# Build vocabulary
vocabulary = set(word for doc in tokenized_docs for word in doc)
vocabulary_list = list(vocabulary)

# Compute raw term frequencies for each document
raw_frequency_vectors = [compute_raw_frequency(doc, vocabulary) for doc in tokenized_docs]

# Create Term-Document Matrix (Raw Frequency)
raw_frequency_matrix = pd.DataFrame(raw_frequency_vectors, columns=vocabulary_list).fillna(0)

# Display results
print("Term-Document Matrix (Raw Frequency):")
print(raw_frequency_matrix)
