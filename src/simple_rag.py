"""
Simple RAG (Retrieval-Augmented Generation) Implementation

This module provides a basic implementation of RAG components:
1. Document retrieval
2. Context augmentation
3. Response generation
"""

import re
from typing import List, Dict, Any
from collections import Counter


class SimpleRetriever:
    """Simple text-based retriever using keyword matching."""
    
    def __init__(self, documents: List[str]):
        """
        Initialize the retriever with a list of documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.processed_docs = self._preprocess_documents()
    
    def _preprocess_documents(self) -> List[Dict[str, Any]]:
        """Preprocess documents for retrieval."""
        processed = []
        for i, doc in enumerate(self.documents):
            # Simple preprocessing: lowercase and split into words
            words = re.findall(r'\b\w+\b', doc.lower())
            processed.append({
                'id': i,
                'text': doc,
                'words': words,
                'word_freq': Counter(words)
            })
        return processed
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_freq = Counter(query_words)
        
        scores = []
        for doc in self.processed_docs:
            # Simple TF-IDF inspired scoring
            score = 0
            for word, freq in query_freq.items():
                if word in doc['word_freq']:
                    score += freq * doc['word_freq'][word]
            scores.append((score, doc))
        
        # Sort by score (first element of tuple) and return top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]


class SimpleGenerator:
    """Simple response generator using template-based generation."""
    
    def __init__(self):
        """Initialize the generator."""
        self.templates = {
            'answer': "Based on the retrieved information: {context}\n\nAnswer: {answer}",
            'summary': "Here's a summary of the relevant information:\n{context}",
            'explanation': "The information suggests that: {context}"
        }
    
    def generate(self, query: str, context: str, style: str = 'answer') -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: Original query
            context: Retrieved context
            style: Generation style ('answer', 'summary', 'explanation')
            
        Returns:
            Generated response
        """
        template = self.templates.get(style, self.templates['answer'])
        
        # Simple answer extraction (in a real system, this would use an LLM)
        sentences = context.split('.')
        relevant_sentences = [s.strip() for s in sentences if any(word in s.lower() 
                                                               for word in query.lower().split())]
        
        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:2]) + '.'
        else:
            answer = "I found some relevant information, but couldn't extract a specific answer."
        
        return template.format(context=context, answer=answer)


class SimpleRAG:
    """Simple RAG system combining retrieval and generation."""
    
    def __init__(self, documents: List[str]):
        """
        Initialize the RAG system.
        
        Args:
            documents: List of document strings
        """
        self.retriever = SimpleRetriever(documents)
        self.generator = SimpleGenerator()
    
    def query(self, question: str, top_k: int = 3, style: str = 'answer') -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            style: Generation style
            
        Returns:
            Dictionary containing response and metadata
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k)
        
        if not retrieved_docs:
            return {
                'response': "I couldn't find any relevant information to answer your question.",
                'retrieved_docs': [],
                'context': ""
            }
        
        # Step 2: Prepare context
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Step 3: Generate response
        response = self.generator.generate(question, context, style)
        
        return {
            'response': response,
            'retrieved_docs': retrieved_docs,
            'context': context
        }


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn.",
        "Natural Language Processing helps computers understand human language.",
        "Vector embeddings represent text as numerical vectors in high-dimensional space.",
        "Information retrieval finds relevant documents from large collections."
    ]
    
    # Initialize RAG system
    rag = SimpleRAG(documents)
    
    # Example query
    question = "What is machine learning?"
    result = rag.query(question)
    
    print(f"Question: {question}")
    print(f"Response: {result['response']}")
    print(f"Retrieved {len(result['retrieved_docs'])} documents") 