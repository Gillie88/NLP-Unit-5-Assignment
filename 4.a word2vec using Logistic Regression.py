from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Load article content from local files
def load_article_from_file(title):
    filepath = f"wiki_articles/{title}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Wikipedia article titles and their class labels
titles = [
    "Robotics", "Aerospace", "Archaeology", "Cryptography", "DNA_replication"
]
labels = [0, 1, 2, 3, 4]  # Each document gets a unique class

# Load and preprocess documents
documents = [load_article_from_file(title.replace(" ", "_")) for title in titles]
tokenized_docs = [doc.lower().split() for doc in documents]

# Train Word2Vec on tokenized corpus
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Convert documents to average dense vectors
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

doc_vectors = np.array([get_doc_vector(tokens, model) for tokens in tokenized_docs])

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)
predictions = classifier.predict(doc_vectors)

# Print classification results
print("\nClassification Report:\n")
print(classification_report(labels, predictions, zero_division=1))
