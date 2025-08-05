#!/usr/bin/env python3
"""
Question-Answering System Example

This script demonstrates a more advanced question-answering system
using RAG techniques with different retrieval and generation strategies.
"""

import sys
import os
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_rag import SimpleRAG
from vector_store import VectorStore
from document_loader import DocumentManager


class QASystem:
    """Advanced Question-Answering system with multiple strategies."""
    
    def __init__(self, documents: List[str]):
        """
        Initialize the QA system.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.simple_rag = SimpleRAG(documents)
        
        # Initialize vector store for advanced retrieval
        try:
            self.vector_store = VectorStore()
            self.vector_store.add_documents(documents)
        except Exception as e:
            print(f"Warning: Vector store initialization failed: {e}")
            self.vector_store = None
    
    def answer_question(self, question: str, strategy: str = "hybrid", top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question using the specified strategy.
        
        Args:
            question: User question
            strategy: Retrieval strategy ('simple', 'vector', 'hybrid')
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        if strategy == "simple":
            return self._simple_retrieval(question, top_k)
        elif strategy == "vector":
            return self._vector_retrieval(question, top_k)
        elif strategy == "hybrid":
            return self._hybrid_retrieval(question, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _simple_retrieval(self, question: str, top_k: int) -> Dict[str, Any]:
        """Use simple keyword-based retrieval."""
        result = self.simple_rag.query(question, top_k=top_k)
        return {
            'answer': result['response'],
            'strategy': 'simple',
            'retrieved_docs': result['retrieved_docs'],
            'context': result['context']
        }
    
    def _vector_retrieval(self, question: str, top_k: int) -> Dict[str, Any]:
        """Use vector-based retrieval."""
        if self.vector_store is None:
            return self._simple_retrieval(question, top_k)
        
        # Get similar documents
        similar_docs = self.vector_store.similarity_search(question, top_k=top_k)
        
        if not similar_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'strategy': 'vector',
                'retrieved_docs': [],
                'context': ""
            }
        
        # Prepare context
        context = "\n\n".join([doc['text'] for doc in similar_docs])
        
        # Generate answer using simple generator
        answer = self.simple_rag.generator.generate(question, context, 'answer')
        
        return {
            'answer': answer,
            'strategy': 'vector',
            'retrieved_docs': similar_docs,
            'context': context
        }
    
    def _hybrid_retrieval(self, question: str, top_k: int) -> Dict[str, Any]:
        """Combine simple and vector retrieval."""
        # Get results from both strategies
        simple_result = self._simple_retrieval(question, top_k // 2)
        vector_result = self._vector_retrieval(question, top_k // 2)
        
        # Combine retrieved documents
        all_docs = simple_result['retrieved_docs'] + vector_result['retrieved_docs']
        
        # Remove duplicates based on document ID
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.get('id', doc.get('text', ''))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        # Prepare combined context
        context = "\n\n".join([doc.get('text', '') for doc in unique_docs])
        
        # Generate answer
        answer = self.simple_rag.generator.generate(question, context, 'answer')
        
        return {
            'answer': answer,
            'strategy': 'hybrid',
            'retrieved_docs': unique_docs,
            'context': context
        }
    
    def get_answer_confidence(self, answer: str, question: str) -> float:
        """
        Calculate confidence score for an answer.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation based on answer length and keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Keyword overlap
        overlap = len(question_words.intersection(answer_words))
        keyword_score = min(overlap / len(question_words), 1.0) if question_words else 0
        
        # Length score (prefer longer answers)
        length_score = min(len(answer) / 200, 1.0)  # Normalize to 200 chars
        
        # Combined score
        confidence = (keyword_score * 0.7) + (length_score * 0.3)
        return confidence


def load_qa_documents():
    """Load documents for the QA system."""
    documents = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. AI systems can perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, perception, and language understanding.",
        
        "Machine Learning is a subset of AI that focuses on developing algorithms and statistical models. These models enable computers to improve their performance on specific tasks through experience, without being explicitly programmed for each task.",
        
        "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks can automatically learn features from raw data, making them particularly effective for image recognition, natural language processing, and speech recognition.",
        
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP combines computational linguistics with machine learning and deep learning to enable machines to understand, interpret, and generate human language.",
        
        "Computer Vision is a field of AI that enables computers to interpret and understand visual information from the world. It involves tasks such as image classification, object detection, facial recognition, and scene understanding.",
        
        "Robotics is an interdisciplinary field that combines computer science, engineering, and other fields to design, construct, and operate robots. Modern robotics often incorporates AI and machine learning for autonomous decision-making.",
        
        "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, computer science, and domain expertise.",
        
        "Big Data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations. It is characterized by Volume (large amount), Velocity (high speed), and Variety (different types) of data.",
        
        "Cloud Computing is the delivery of computing services over the internet, including servers, storage, databases, networking, software, and analytics. It provides on-demand access to shared computing resources.",
        
        "The Internet of Things (IoT) refers to the network of physical objects embedded with sensors, software, and other technologies to connect and exchange data with other devices and systems over the internet."
    ]
    return documents


def interactive_qa_demo():
    """Run an interactive QA demonstration."""
    print("🤖 Advanced Question-Answering System")
    print("=" * 60)
    
    # Load documents
    print("Loading documents...")
    documents = load_qa_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Initialize QA system
    print("Initializing QA system...")
    qa_system = QASystem(documents)
    print("✅ QA system ready!")
    
    print("\n" + "=" * 60)
    print("💡 Ask questions about AI, machine learning, and related topics.")
    print("Available strategies: simple, vector, hybrid")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the QA system!")
                break
            
            if not question:
                continue
            
            # Get strategy preference
            strategy = input("🎯 Strategy (simple/vector/hybrid) [hybrid]: ").strip().lower()
            if not strategy:
                strategy = "hybrid"
            
            if strategy not in ['simple', 'vector', 'hybrid']:
                strategy = "hybrid"
            
            # Process question
            print(f"\n🔍 Searching using {strategy} strategy...")
            result = qa_system.answer_question(question, strategy=strategy, top_k=3)
            
            # Calculate confidence
            confidence = qa_system.get_answer_confidence(result['answer'], question)
            
            # Display results
            print(f"\n📝 Answer (confidence: {confidence:.2f}):")
            print(result['answer'])
            
            print(f"\n📚 Retrieved {len(result['retrieved_docs'])} documents:")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                doc_text = doc.get('text', doc.get('content', ''))
                print(f"  {i}. {doc_text[:100]}...")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try another question.")


def compare_strategies():
    """Compare different QA strategies."""
    print("🔬 Strategy Comparison")
    print("=" * 60)
    
    # Load documents and initialize system
    documents = load_qa_documents()
    qa_system = QASystem(documents)
    
    # Test questions
    test_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is deep learning?",
        "Explain natural language processing",
        "What is computer vision?"
    ]
    
    strategies = ['simple', 'vector', 'hybrid']
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        for strategy in strategies:
            result = qa_system.answer_question(question, strategy=strategy, top_k=2)
            confidence = qa_system.get_answer_confidence(result['answer'], question)
            
            print(f"\n🎯 {strategy.upper()} Strategy:")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Retrieved docs: {len(result['retrieved_docs'])}")
            print(f"   Answer: {result['answer'][:150]}...")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the QA system demo."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare':
            compare_strategies()
        else:
            print("Usage: python qa_system.py [--compare]")
    else:
        interactive_qa_demo()


if __name__ == "__main__":
    main() 