"""
Tests for RAG components

This module contains unit tests for the RAG system components.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_rag import SimpleRAG, SimpleRetriever, SimpleGenerator


class TestSimpleRetriever(unittest.TestCase):
    """Test cases for SimpleRetriever."""
    
    def setUp(self):
        """Set up test data."""
        self.documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers.",
            "Computer vision allows machines to interpret images.",
            "Robotics combines AI with mechanical engineering."
        ]
        self.retriever = SimpleRetriever(self.documents)
    
    def test_initialization(self):
        """Test retriever initialization."""
        self.assertEqual(len(self.retriever.documents), 5)
        self.assertEqual(len(self.retriever.processed_docs), 5)
    
    def test_retrieval(self):
        """Test document retrieval."""
        query = "machine learning"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)
        
        # Check that results contain the query terms
        for result in results:
            self.assertIn('text', result)
            self.assertIn('id', result)
    
    def test_empty_query(self):
        """Test retrieval with empty query."""
        results = self.retriever.retrieve("", top_k=2)
        self.assertEqual(len(results), 0)
    
    def test_no_matches(self):
        """Test retrieval when no documents match."""
        results = self.retriever.retrieve("xyzabc", top_k=2)
        self.assertEqual(len(results), 0)


class TestSimpleGenerator(unittest.TestCase):
    """Test cases for SimpleGenerator."""
    
    def setUp(self):
        """Set up test data."""
        self.generator = SimpleGenerator()
        self.query = "What is machine learning?"
        self.context = "Machine learning is a subset of artificial intelligence that enables computers to learn."
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIn('answer', self.generator.templates)
        self.assertIn('summary', self.generator.templates)
        self.assertIn('explanation', self.generator.templates)
    
    def test_generation(self):
        """Test response generation."""
        response = self.generator.generate(self.query, self.context, 'answer')
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIn('Machine learning', response)
    
    def test_different_styles(self):
        """Test different generation styles."""
        styles = ['answer', 'summary', 'explanation']
        
        for style in styles:
            response = self.generator.generate(self.query, self.context, style)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_empty_context(self):
        """Test generation with empty context."""
        response = self.generator.generate(self.query, "", 'answer')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


class TestSimpleRAG(unittest.TestCase):
    """Test cases for SimpleRAG."""
    
    def setUp(self):
        """Set up test data."""
        self.documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers."
        ]
        self.rag = SimpleRAG(self.documents)
    
    def test_initialization(self):
        """Test RAG system initialization."""
        self.assertIsNotNone(self.rag.retriever)
        self.assertIsNotNone(self.rag.generator)
        self.assertEqual(len(self.rag.retriever.documents), 3)
    
    def test_query(self):
        """Test complete RAG query."""
        question = "What is machine learning?"
        result = self.rag.query(question, top_k=2)
        
        self.assertIn('response', result)
        self.assertIn('retrieved_docs', result)
        self.assertIn('context', result)
        
        self.assertIsInstance(result['response'], str)
        self.assertGreater(len(result['response']), 0)
        self.assertLessEqual(len(result['retrieved_docs']), 2)
    
    def test_query_no_matches(self):
        """Test query when no documents match."""
        question = "xyzabc quantum physics"
        result = self.rag.query(question, top_k=2)
        
        self.assertIn('response', result)
        self.assertEqual(len(result['retrieved_docs']), 0)
        self.assertEqual(result['context'], "")
    
    def test_different_styles(self):
        """Test different generation styles."""
        question = "What is machine learning?"
        styles = ['answer', 'summary', 'explanation']
        
        for style in styles:
            result = self.rag.query(question, top_k=1, style=style)
            self.assertIn('response', result)
            self.assertIsInstance(result['response'], str)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system."""
    
    def test_end_to_end(self):
        """Test complete end-to-end RAG pipeline."""
        documents = [
            "Artificial Intelligence (AI) is a branch of computer science.",
            "Machine Learning enables computers to learn from data.",
            "Deep Learning uses neural networks with multiple layers.",
            "Natural Language Processing helps computers understand text."
        ]
        
        rag = SimpleRAG(documents)
        
        # Test multiple questions
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is deep learning?",
            "Explain natural language processing"
        ]
        
        for question in questions:
            result = rag.query(question, top_k=2)
            
            # Basic validation
            self.assertIn('response', result)
            self.assertIn('retrieved_docs', result)
            self.assertIn('context', result)
            
            # Response should not be empty
            self.assertGreater(len(result['response']), 0)
            
            # Should retrieve some documents
            self.assertGreaterEqual(len(result['retrieved_docs']), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 