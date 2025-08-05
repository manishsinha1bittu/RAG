#!/usr/bin/env python3
"""
Basic RAG Example

This script demonstrates a simple RAG (Retrieval-Augmented Generation) system
using the implemented components.
"""

import sys
import os
from typing import List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_rag import SimpleRAG
from vector_store import VectorStore


def load_sample_documents() -> List[str]:
    """Load sample documents from the data directory."""
    docs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_docs.txt')
    
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split documents by double newlines
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        print(f"Loaded {len(documents)} documents")
        return documents
    
    except FileNotFoundError:
        print(f"Error: Could not find {docs_path}")
        return []


def demonstrate_simple_rag():
    """Demonstrate the simple RAG system."""
    print("=== Simple RAG Demonstration ===\n")
    
    # Load documents
    documents = load_sample_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    # Initialize RAG system
    rag = SimpleRAG(documents)
    
    # Example questions
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain neural networks",
        "What is deep learning used for?"
    ]
    
    print("Sample Questions and Answers:\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        
        # Get answer using simple RAG
        result = rag.query(question, top_k=2)
        
        print(f"Answer: {result['response']}")
        print(f"Retrieved {len(result['retrieved_docs'])} documents")
        print("-" * 80)
        print()


def demonstrate_vector_store():
    """Demonstrate the vector store functionality."""
    print("=== Vector Store Demonstration ===\n")
    
    # Load documents
    documents = load_sample_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        print("Adding documents to vector store...")
        vector_store.add_documents(documents)
        print("Documents added successfully!\n")
        
        # Example queries
        queries = [
            "What is AI?",
            "Machine learning algorithms",
            "Neural networks and deep learning"
        ]
        
        for query in queries:
            print(f"Query: {query}")
            results = vector_store.similarity_search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  Result {i} (Score: {result['score']:.3f}):")
                print(f"    {result['text'][:100]}...")
            print()
            
    except Exception as e:
        print(f"Error with vector store: {e}")
        print("Make sure you have installed sentence-transformers: pip install sentence-transformers")


def main():
    """Main function to run the demonstrations."""
    print("RAG System Basic Example\n")
    print("This example demonstrates:")
    print("1. Simple RAG with keyword-based retrieval")
    print("2. Vector store with semantic search")
    print()
    
    # Demonstrate simple RAG
    demonstrate_simple_rag()
    
    # Demonstrate vector store
    demonstrate_vector_store()
    
    print("Demonstration complete!")


if __name__ == "__main__":
    main() 