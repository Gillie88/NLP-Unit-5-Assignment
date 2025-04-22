import wikipediaapi
from pathlib import Path

# Prepare directory for storing articles
corpus_path = Path("wiki_articles")
corpus_path.mkdir(exist_ok=True)

# Initialize Wikipedia interface
wiki_api = wikipediaapi.Wikipedia(
    language='en',
    user_agent='YourApp/1.0 (https://yourwebsite.com)'
)

# Selected topics for the corpus
documents = [
    "Robotics",
    "Aerospace",
    "Archaeology",
    "cryptography",
    "DNA replication"
]

# Loop to retrieve and save articles
for doc_title in documents:
    wiki_page = wiki_api.page(doc_title)
    
    if wiki_page.exists():
        file_path = corpus_path / f"{doc_title.replace(' ', '_')}.txt"
        file_path.write_text(wiki_page.text, encoding='utf-8')
        print(f"[✓] Document saved: {file_path.name}")
    else:
        print(f"[!] Skipped (not found): {doc_title}")
