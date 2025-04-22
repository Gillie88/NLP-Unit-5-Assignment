import math
from collections import Counter
from pathlib import Path

# Function to load Wikipedia article text from a local file
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Calculate the Cosine Similarity
def cosine_similarity(vec1, vec2, vocab):
    # Get the dot product of the two vectors
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vocab)
    vec1Length = math.sqrt(sum(vec1.get(term, 0)**2 for term in vocab))
    vec2Length = math.sqrt(sum(vec2.get(term, 0)**2 for term in vocab))

    if vec1Length == 0 or vec2Length == 0:
        return 0.0

    return dot_product / (vec1Length * vec2Length)

# Selected topics for the corpus (matching saved document filenames)
documents = [
    "Robotics",
    "Aerospace",
    "Archaeology",
    "cryptography",
    "DNA_replication"
]

# Load documents from local folder
tokenized_docs = {doc: load_article_from_file(doc).lower().split() for doc in documents}

# Prepare vocabulary (unique words from all documents)
vocab = set(word for doc_tokens in tokenized_docs.values() for word in doc_tokens)

# Convert documents into term frequency vectors (using Counter)
tf_vectors = {doc: Counter(doc_tokens) for doc, doc_tokens in tokenized_docs.items()}

# Variables to track the most similar documents
most_similar_pair = None
max_similarity_score = 0

# Compare all documents to each other and calculate cosine similarity
print("\nPairwise Cosine Similarities:\n")
for doc1 in documents:
    for doc2 in documents:
        if doc1 != doc2:  # Skip self-comparison
            similarity_score = cosine_similarity(tf_vectors[doc1], tf_vectors[doc2], vocab)
            print(f"Cosine similarity between '{doc1}' and '{doc2}': {similarity_score:.4f}")
            print("--------------------------------------------------------------------------")
            
            # Update the most similar pair if the current similarity score is higher
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                most_similar_pair = (doc1, doc2)

# Print the most similar documents
if most_similar_pair:
    print(f"\n🔍 The most similar documents are '{most_similar_pair[0]}' and '{most_similar_pair[1]}' with a cosine similarity of {max_similarity_score:.4f}")
