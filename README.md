# RAG (Retrieval-Augmented Generation) System

A comprehensive implementation of a Retrieval-Augmented Generation (RAG) system with multiple retrieval strategies and generation capabilities.

## Overview

This project implements a complete RAG system with the following components:

- **Document Loading**: Utilities for loading and preprocessing documents
- **Vector Storage**: FAISS-based vector store with sentence transformers
- **Retrieval**: Multiple retrieval strategies (keyword-based, vector-based, hybrid)
- **Generation**: Template-based response generation
- **QA System**: Advanced question-answering system with multiple strategies

## Project Structure

```
RAG/
├── src/                    # Core RAG components
│   ├── __init__.py        # Package initialization
│   ├── document_loader.py # Document loading utilities
│   ├── vector_store.py    # Vector storage implementation
│   └── simple_rag.py      # Basic RAG implementation
├── examples/              # Example scripts
│   ├── basic_rag_example.py # Basic RAG demonstration
│   └── qa_system.py       # Advanced QA system
├── data/                  # Data directory
│   ├── sample_docs.txt    # Sample documents for testing
│   └── knowledge_base/    # Knowledge base directory
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import sentence_transformers; print('Installation successful!')"
   ```

## Quick Start

### Basic RAG Example

Run the basic RAG demonstration:

```bash
python examples/basic_rag_example.py
```

This will:
- Load sample documents about AI and machine learning
- Demonstrate simple keyword-based retrieval
- Show vector store functionality with semantic search

### Advanced QA System

Run the advanced question-answering system:

```bash
python examples/qa_system.py
```

This provides an interactive interface with multiple retrieval strategies.

## Core Components

### 1. Vector Store (`src/vector_store.py`)

Provides vector storage and similarity search using sentence transformers and FAISS:

```python
from src.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Add documents
documents = ["Document 1", "Document 2", "Document 3"]
vector_store.add_documents(documents)

# Search for similar documents
results = vector_store.similarity_search("your query", top_k=5)
```

### 2. Simple RAG (`src/simple_rag.py`)

Basic RAG implementation with retrieval and generation:

```python
from src.simple_rag import SimpleRAG

# Initialize RAG system
rag = SimpleRAG(documents)

# Query the system
result = rag.query("What is machine learning?", top_k=3)
print(result['response'])
```

### 3. QA System (`examples/qa_system.py`)

Advanced question-answering system with multiple strategies:

```python
from examples.qa_system import QASystem

# Initialize QA system
qa = QASystem(documents)

# Answer questions with different strategies
answer = qa.answer_question("What is AI?", strategy="hybrid")
```

## Features

### Retrieval Strategies

1. **Simple Retrieval**: Keyword-based matching using TF-IDF inspired scoring
2. **Vector Retrieval**: Semantic search using sentence embeddings
3. **Hybrid Retrieval**: Combination of both approaches for better results

### Generation Styles

- **Answer**: Direct answer generation
- **Summary**: Context summarization
- **Explanation**: Detailed explanations

### Vector Store Features

- **FAISS Integration**: Fast similarity search
- **Sentence Transformers**: High-quality embeddings
- **Persistence**: Save and load vector stores
- **Multiple Models**: Support for different embedding models

## Usage Examples

### Adding Your Own Documents

```python
# Load your documents
with open('your_documents.txt', 'r') as f:
    documents = f.read().split('\n\n')

# Initialize RAG system
rag = SimpleRAG(documents)

# Query your documents
result = rag.query("Your question here")
```

### Using Different Embedding Models

```python
# Use a different sentence transformer model
vector_store = VectorStore(model_name="all-mpnet-base-v2")
```

### Customizing Retrieval

```python
# Adjust number of retrieved documents
result = rag.query("Question", top_k=5)

# Use different generation style
result = rag.query("Question", style="summary")
```

## Configuration

### Environment Variables

- `SENTENCE_TRANSFORMERS_HOME`: Cache directory for models
- `FAISS_CACHE_DIR`: Cache directory for FAISS indices

### Model Options

Available sentence transformer models:
- `all-MiniLM-L6-v2` (default, fast, 384 dimensions)
- `all-mpnet-base-v2` (high quality, 768 dimensions)
- `all-MiniLM-L12-v2` (balanced, 384 dimensions)

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Performance Considerations

- **Memory Usage**: Vector stores can be memory-intensive with large document collections
- **Processing Speed**: Sentence transformers require initial model download
- **Storage**: FAISS indices can be saved to disk for reuse

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Download**: First run may take time to download models
3. **Memory Issues**: Consider using smaller models for large document collections

### Getting Help

- Check the example scripts for usage patterns
- Review the docstrings in source files
- Ensure your Python environment has sufficient memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Sentence Transformers library for embeddings
- FAISS for efficient similarity search
- The RAG research community for foundational concepts 