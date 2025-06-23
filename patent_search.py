import requests
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"

class PatentSearch:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.document_store = []

    def get_sample_patents(self):
        """Get sample patent data for demonstration"""
        sample_patents = [
            {
                "patent_id": "US10123456",
                "title": "Machine Learning System for Healthcare Diagnosis",
                "abstract": "A computer-implemented method for diagnosing medical conditions using machine learning algorithms trained on large datasets of patient information, medical images, and clinical outcomes."
            },
            {
                "patent_id": "US10123457", 
                "title": "Artificial Intelligence for Drug Discovery",
                "abstract": "An AI system that analyzes molecular structures and predicts drug efficacy using deep learning neural networks and computational chemistry methods."
            },
            {
                "patent_id": "US10123458",
                "title": "Blockchain-Based Medical Records System",
                "abstract": "A decentralized system for storing and sharing medical records using blockchain technology to ensure data security and patient privacy."
            },
            {
                "patent_id": "US10123459",
                "title": "IoT Device for Remote Patient Monitoring",
                "abstract": "An Internet of Things device that continuously monitors vital signs and transmits data to healthcare providers for real-time patient care."
            },
            {
                "patent_id": "US10123460",
                "title": "Natural Language Processing for Medical Documentation",
                "abstract": "A system that uses NLP to automatically generate medical reports from doctor-patient conversations and clinical notes."
            },
            {
                "patent_id": "US10123461",
                "title": "Computer Vision for Medical Imaging Analysis",
                "abstract": "Deep learning algorithms for analyzing X-rays, MRIs, and CT scans to detect abnormalities and assist radiologists in diagnosis."
            },
            {
                "patent_id": "US10123462",
                "title": "Predictive Analytics for Hospital Resource Management",
                "abstract": "Machine learning models that predict patient admission rates and optimize hospital staffing and resource allocation."
            },
            {
                "patent_id": "US10123463",
                "title": "Telemedicine Platform with AI Assistant",
                "abstract": "A virtual healthcare platform that uses AI chatbots to triage patients and connect them with appropriate medical specialists."
            },
            {
                "patent_id": "US10123464",
                "title": "Genomic Data Analysis Using Machine Learning",
                "abstract": "AI algorithms for analyzing genetic sequences to identify disease markers and predict individual health risks."
            },
            {
                "patent_id": "US10123465",
                "title": "Wearable Technology for Health Monitoring",
                "abstract": "Smart wearable devices that track physical activity, sleep patterns, and health metrics using embedded sensors and AI."
            }
        ]
        return sample_patents

    def fetch_patents(self, query_params=None):
        """Fetch patents - using sample data for demonstration"""
        print("Using sample patent data for demonstration...")
        return self.get_sample_patents()

    def process_patents(self, patents):
        """Process patents into documents"""
        documents = []
        for patent in patents:
            text = f"{patent.get('title', '')} {patent.get('abstract', '')}"
            if text.strip():  # Only add non-empty documents
                documents.append({
                    'content': text,
                    'patent_id': patent.get('patent_id', 'Unknown'),
                    'title': patent.get('title', ''),
                    'abstract': patent.get('abstract', '')
                })
        return documents

    def index_patents(self, documents):
        """Index patents using FAISS"""
        if not documents:
            print("No documents to index")
            return
        
        # Extract text content for embedding
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Store documents and embeddings
        self.documents = documents
        self.embeddings = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Indexed {len(documents)} patents successfully")

    def search(self, query, top_k=5):
        """Search patents using vector similarity"""
        if self.index is None or len(self.documents) == 0:
            print("No patents indexed. Please index patents first.")
            return pd.DataFrame()
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'rank': i + 1,
                    'patent_id': doc['patent_id'],
                    'title': doc['title'],
                    'score': float(score),
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                })
        
        return pd.DataFrame(results)

    def test_search_queries(self):
        """Test the search system with various queries"""
        print("\n=== Testing Patent Search System ===")
        
        test_cases = [
            {
                "query": "machine learning healthcare",
                "expected_keywords": ["machine learning", "healthcare", "diagnosis"],
                "description": "Should find ML healthcare diagnosis patents"
            },
            {
                "query": "artificial intelligence drug discovery",
                "expected_keywords": ["AI", "drug", "discovery", "molecular"],
                "description": "Should find AI drug discovery patents"
            },
            {
                "query": "blockchain medical records",
                "expected_keywords": ["blockchain", "medical", "records", "privacy"],
                "description": "Should find blockchain medical records patents"
            },
            {
                "query": "IoT patient monitoring",
                "expected_keywords": ["IoT", "patient", "monitoring", "vital"],
                "description": "Should find IoT monitoring patents"
            },
            {
                "query": "natural language processing medical",
                "expected_keywords": ["NLP", "medical", "documentation", "reports"],
                "description": "Should find NLP medical documentation patents"
            },
            {
                "query": "computer vision medical imaging",
                "expected_keywords": ["computer vision", "medical", "imaging", "X-ray"],
                "description": "Should find computer vision medical imaging patents"
            },
            {
                "query": "predictive analytics hospital",
                "expected_keywords": ["predictive", "analytics", "hospital", "resource"],
                "description": "Should find predictive analytics hospital patents"
            },
            {
                "query": "telemedicine AI assistant",
                "expected_keywords": ["telemedicine", "AI", "assistant", "chatbot"],
                "description": "Should find telemedicine AI assistant patents"
            },
            {
                "query": "genomic data analysis",
                "expected_keywords": ["genomic", "data", "analysis", "genetic"],
                "description": "Should find genomic data analysis patents"
            },
            {
                "query": "wearable health monitoring",
                "expected_keywords": ["wearable", "health", "monitoring", "sensors"],
                "description": "Should find wearable health monitoring patents"
            },
            {
                "query": "cybersecurity medical devices",
                "expected_keywords": ["cybersecurity", "medical", "devices"],
                "description": "Should find few or no results (not in sample data)"
            },
            {
                "query": "quantum computing healthcare",
                "expected_keywords": ["quantum", "computing", "healthcare"],
                "description": "Should find few or no results (not in sample data)"
            }
        ]
        
        total_tests = len(test_cases)
        passed_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}/{total_tests}: {test_case['description']} ---")
            print(f"Query: '{test_case['query']}'")
            
            results = self.search(test_case['query'], top_k=3)
            
            if not results.empty:
                print("Top Results:")
                for _, row in results.iterrows():
                    print(f"  {row['rank']}. {row['patent_id']} - {row['title']} (Score: {row['score']:.3f})")
                
                # Check if results contain expected keywords
                top_result_content = results.iloc[0]['content'].lower()
                expected_keywords = [kw.lower() for kw in test_case['expected_keywords']]
                found_keywords = [kw for kw in expected_keywords if kw in top_result_content]
                
                if found_keywords:
                    print(f"✓ Found expected keywords: {found_keywords}")
                    passed_tests += 1
                else:
                    print(f"✗ Missing expected keywords. Found: {found_keywords}")
                    print(f"  Top result content: {top_result_content[:100]}...")
            else:
                print("No results found")
                # For queries that shouldn't have results, this might be expected
                if "not in sample data" in test_case['description']:
                    print("✓ Expected no results for this query")
                    passed_tests += 1
                else:
                    print("✗ Unexpected: No results found")
        
        print(f"\n=== Test Summary ===")
        print(f"Passed: {passed_tests}/{total_tests} tests ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! The search system is working correctly.")
        else:
            print("⚠️  Some tests failed. Review the results above.")

if __name__ == "__main__":
    # Example usage
    searcher = PatentSearch()
    
    print("Initializing Patent Search System...")
    # Fetch sample patents
    patents = searcher.fetch_patents()
    
    if patents:
        print(f"Loaded {len(patents)} sample patents")
        
        # Index the patents
        documents = searcher.process_patents(patents)
        searcher.index_patents(documents)
        
        # Run automated tests
        searcher.test_search_queries()
    else:
        print("No patents loaded. Check your data source.")
