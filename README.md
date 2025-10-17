# 🤖 RAG Chatbot Pipeline

Pipeline de traitement de documents optimisé pour chatbot RAG (Retrieval-Augmented Generation). Transforme vos documents en chunks intelligents avec embeddings vectoriels pour une recherche sémantique ultra-rapide.

## ✨ Fonctionnalités

- **🔄 Multi-formats**: Support TXT, JSON, DOCX, PDF, DOC
- **✂️ Chunking intelligent**: Découpage optimal en 40-60 tokens
- **🧠 Embeddings vectoriels**: Génération d'embeddings avec sentence-transformers
- **🎯 Détection de sections**: Identification automatique des structures de document
- **🚫 Anti-duplication**: Élimination complète des redondances
- **⚡ Optimisé pour la vitesse**: Traitement par lots et pipeline efficace

## 📁 Structure du Projet

```
chatbot_rag/
├── api/
│   ├── __init__.py
│   ├── main.py              # API FastAPI principale  
│   ├── models.py            # Modèles Pydantic
│   ├── auth.py              # Auth avec Supabase
│   ├── database.py          # Interface Supabase
│   └── runtime_llm.py        # choix entre cloud or local  
│   └── config.py 
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration centralisée
│   └── ollama_config.py        
├── core/
│   ├── __init__.py
│   ├── document_loader.py   # Chargement multi-formats
│   ├── text_processor.py    # Traitement de texte
│   ├── chunker.py          # Découpage en chunks
│   ├── embedder.py         # Génération d'embeddings
│   └── section_detector.py # Détection de sections
├── indexing/
│   ├── __init__.py
│   ├── faiss_indexer.py      # Création et gestion index FAISS
│   └── index_builder.py      # Construction des index
├── rag/                      # NOUVEAU DOSSIER RAG
│   ├── __init__.py
│   ├── retriever.py          # Récupération de contexte
│   ├── generator.py          # Génération de réponses
│   ├── rag_engine.py         # Engine RAG complet
│   └── helpers.py            # Fonctions utilitaires
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   └── text_utils.py
├── pipeline/
│   ├── __init__.py
│   └── rag_pipeline.py     # Pipeline principal
├── data/
│   ├── raw/                  # Documents d'entrée
│   ├── processed/            # Chunks et embeddings JSON
│   ├── index/                # Index FAISS locaux
│   │   ├── faiss.index
│   │   ├── idmap.json
│   │   └── metadata.json
│   └── runtime/    
├── main.py               # Point d'entrée
├── rag_server.py             
├── requirements.txt      # Dépendances
└── README.md            # Ce fichier
```

## 🚀 Installation

### 1. Clonage et Setup

```bash
git clone <votre-repo>
cd chatbot_rag
```

### 2. Installation des Dépendances

```bash
# Installation complète
pip install -r requirements.txt

# Installation minimale (embeddings factices)
pip install numpy torch
```

### 3. Vérification des Dépendances

```bash
python main.py --check-deps
```

## 📖 Utilisation

### Utilisation Basique

```bash
# Traitement avec dossiers par défaut
python main.py

# Spécifier les dossiers
python main.py --input ./mes_docs --output ./resultats

# Mode verbose
python main.py --verbose
```

### Utilisation en Code Python

```python
from pipeline.rag_pipeline import RAGPipeline
from pathlib import Path

# Initialisation du pipeline
pipeline = RAGPipeline()

# Traitement complet
result = pipeline.run_full_pipeline(
    documents_path=Path("./data/raw"),
    output_path=Path("./data/processed")
)

if result['success']:
    print(f"Chunks créés: {result['stats']['chunks_created']}")
    print(f"Embeddings: {result['stats']['embeddings_generated']}")
```

### Traitement d'un Document Unique

```python
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.embedder import ChunkEmbedder

# Chargement
loader = DocumentLoader()
document = loader.load_document(Path("mon_doc.pdf"))

# Chunking
chunker = DocumentChunker()
chunks = chunker.chunk_document(document)

# Embeddings
embedder = ChunkEmbedder()
embeddings = embedder.create_embeddings(chunks)
```

## ⚙️ Configuration

Modifiez `config/settings.py` pour personnaliser:

```python
# Paramètres de tokens
TARGET_TOKENS = 50      # Taille cible des chunks
MAX_TOKENS = 60        # Taille maximale
MIN_TOKENS = 40        # Taille minimale

# Modèle d'embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Chemins
DOCUMENTS_PATH = "data/raw"
OUTPUT_PATH = "data/processed"
```

## 📊 Format de Sortie

### Chunks (chunks.json)
```json
[
  {
    "id": 1,
    "content": "Contenu du chunk...",
    "source": "document.pdf",
    "section": "Introduction"
  }
]
```

### Embeddings (embeddings.json)
```json
{
  "1": {
    "embedding": [0.1, -0.2, 0.3, ...],
    "content": "Contenu du chunk...",
    "source": "document.pdf",
    "section": "Introduction"
  }
}
```

## 🧪 Tests

```bash
# Exécution des tests
pytest tests/

# Tests avec couverture
pytest tests/ --cov=core --cov=utils --cov=pipeline
```

## 🔧 Développement

### Ajout d'un Nouveau Format

1. Étendre `DocumentLoader` dans `core/document_loader.py`
2. Ajouter l'extension à `SUPPORTED_EXTENSIONS`
3. Implémenter la méthode de lecture

### Personnalisation du Chunking

Modifiez `DocumentChunker` dans `core/chunker.py` pour:
- Nouveaux algorithmes de découpage
- Critères de qualité personnalisés
- Méthodes de détection de duplication

### Nouveaux Modèles d'Embeddings

Étendre `EmbeddingModel` dans `core/embedder.py`:
- Support d'autres modèles Hugging Face
- Embeddings OpenAI API
- Modèles personnalisés

## 🚨 Résolution de Problèmes

### Erreur "sentence-transformers not found"
```bash
pip install sentence-transformers
```

### Erreur DOCX/PDF
```bash
pip install python-docx PyMuPDF
```

### Mémoire insuffisante
- Réduire `EMBEDDING_BATCH_SIZE`
- Utiliser un modèle plus petit
- Traiter moins de documents à la fois

### Chunks trop petits/grands
- Ajuster `TARGET_TOKENS`, `MIN_TOKENS`, `MAX_TOKENS`
- Modifier `TOKENS_PER_WORD` pour votre langue

## 📈 Performance

### Optimisations Recommandées

1. **GPU**: Utiliser CUDA pour les embeddings
2. **Batch Size**: Ajuster selon votre RAM
3. **Modèle**: Choisir selon précision vs vitesse
4. **Multiprocessing**: Pour traitement de gros volumes

### Benchmarks Typiques

| Documents | Chunks | Temps | RAM |
|-----------|--------|-------|-----|
| 10 PDF    | ~500   | 30s   | 2GB |
| 50 DOCX   | ~2000  | 2min  | 4GB |
| 100 TXT   | ~5000  | 5min  | 8GB |

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - voir LICENSE pour les détails.

## 🔗 Intégration Chatbot

Ce pipeline génère les données optimisées pour:
- **Recherche vectorielle**: Utiliser les embeddings pour similarity search
- **Context retrieval**: Récupérer les chunks pertinents
- **Response generation**: Alimenter votre modèle de language

Exemple d'intégration:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_similar_chunks(query_embedding, embeddings_dict, top_k=5):
    similarities = []
    for chunk_id, data in embeddings_dict.items():
        sim = cosine_similarity([query_embedding], [data['embedding']])[0][0]
        similarities.append((chunk_id, sim, data['content']))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

---

🎉 **Votre pipeline RAG est maintenant prêt pour alimenter votre chatbot intelligent !**