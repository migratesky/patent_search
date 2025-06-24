import requests
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json
import gzip
import os
import time
import zipfile
from urllib.parse import urljoin
from tqdm import tqdm
import logging

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = "patent_data"
LOCAL_DATA_FILE = "g_detail_desc_text_2024.tsv.zip"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatentSearch:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.document_store = []
        self.data_dir = DATA_DIR
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

    def load_local_patent_data(self, max_patents=None, filter_keywords=None, sample_fraction=1.0):
        """Load patent data from local ZIP file"""
        if not os.path.exists(LOCAL_DATA_FILE):
            logger.error(f"Local data file {LOCAL_DATA_FILE} not found")
            return []
        
        logger.info(f"Loading patent data from {LOCAL_DATA_FILE}")
        
        try:
            # Read the ZIP file directly
            with zipfile.ZipFile(LOCAL_DATA_FILE, 'r') as zip_file:
                # Get the TSV file name from the ZIP
                tsv_filename = zip_file.namelist()[0]
                logger.info(f"Found TSV file: {tsv_filename}")
                
                # Read in chunks to manage memory
                chunk_size = 10000
                all_patents = []
                total_loaded = 0
                
                # Read the TSV file from the ZIP
                with zip_file.open(tsv_filename) as tsv_file:
                    # Read header first
                    header = tsv_file.readline().decode('utf-8').strip().split('\t')
                    logger.info(f"Columns: {header}")
                    
                    # Read data in chunks
                    for chunk_num, chunk_df in enumerate(pd.read_csv(tsv_file, 
                                                                   sep='\t', 
                                                                   chunksize=chunk_size,
                                                                   names=header,
                                                                   low_memory=False)):
                        
                        # Clean column names (remove quotes)
                        chunk_df.columns = [col.strip('"') for col in chunk_df.columns]
                        
                        # Filter out rows with missing essential data
                        chunk_df = chunk_df.dropna(subset=['description_text'])
                        
                        # Apply keyword filtering if specified
                        if filter_keywords:
                            mask = chunk_df['description_text'].str.contains('|'.join(filter_keywords), 
                                                                           case=False, na=False)
                            chunk_df = chunk_df[mask]
                        
                        # Convert to list of dictionaries
                        patents = chunk_df.to_dict('records')
                        all_patents.extend(patents)
                        total_loaded += len(patents)
                        
                        logger.info(f"Loaded chunk {chunk_num + 1}: {len(patents)} patents (Total: {total_loaded})")
                        
                        # Check if we've reached the limit
                        if max_patents and total_loaded >= max_patents:
                            logger.info(f"Reached maximum patents limit ({max_patents})")
                            break
                
                # Apply sampling if specified
                if sample_fraction < 1.0:
                    sample_size = int(len(all_patents) * sample_fraction)
                    all_patents = np.random.choice(all_patents, sample_size, replace=False).tolist()
                    logger.info(f"Sampled {len(all_patents)} patents from {total_loaded} total")
                
                # Limit to max_patents if specified
                if max_patents:
                    all_patents = all_patents[:max_patents]
                
                logger.info(f"Total patents loaded: {len(all_patents)}")
                return all_patents
                
        except Exception as e:
            logger.error(f"Error loading patent data: {e}")
            return []

    def fetch_patents(self, query_params=None):
        """Fetch patents from local data file with smart defaults"""
        # Default parameters for moderate dataset
        max_patents = 5000  # 5K patents for moderate dataset
        
        # Technology-focused keywords for better relevance
        filter_keywords = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'computer vision', 'natural language processing', 'robotics', 'automation',
            'blockchain', 'cybersecurity', 'cloud computing', 'big data',
            'internet of things', 'IoT', 'mobile', 'software', 'algorithm',
            'predictive', 'analytics', 'optimization', 'simulation', 'system',
            'method', 'apparatus', 'device', 'technology', 'digital'
        ]
        
        # Load patent data with filtering
        logger.info("Loading and filtering patent data from local file...")
        patents = self.load_local_patent_data(
            max_patents=max_patents,
            filter_keywords=filter_keywords,
            sample_fraction=0.3  # Use 30% of available data for efficiency
        )
        
        return patents

    def process_patents(self, patents):
        """Process patents into documents with enhanced text processing"""
        documents = []
        
        logger.info(f"Processing {len(patents)} patents...")
        
        for i, patent in enumerate(tqdm(patents, desc="Processing patents")):
            patent_id = patent.get('patent_id', f'Unknown_{i}')
            description = patent.get('description_text', '').strip()
            
            # Skip patents with insufficient text
            if len(description) < 100:
                continue
            
            # Clean and normalize text
            # Remove common patent boilerplate
            description = description.replace('DETAILED DESCRIPTION', '').strip()
            description = description.replace('BACKGROUND OF THE INVENTION', '').strip()
            description = description.replace('SUMMARY OF THE INVENTION', '').strip()
            
            # Take first 2000 characters to keep it manageable
            description = description[:2000]
            
            # Clean whitespace
            description = ' '.join(description.split())
            
            if description.strip():
                documents.append({
                    'content': description,
                    'patent_id': patent_id,
                    'title': f"Patent {patent_id}",  # We don't have titles, so use patent ID
                    'abstract': description[:500] + "..." if len(description) > 500 else description,
                    'patent_date': '2024'  # Approximate date from filename
                })
        
        logger.info(f"Successfully processed {len(documents)} documents")
        return documents

    def index_patents(self, documents):
        """Index patents using FAISS with memory-efficient processing"""
        if not documents:
            logger.warning("No documents to index")
            return
        
        # Extract text content for embedding
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings in batches to manage memory
        batch_size = 1000
        all_embeddings = []
        
        logger.info(f"Generating embeddings for {len(texts)} documents in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Store documents and embeddings
        self.documents = documents
        self.embeddings = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Successfully indexed {len(documents)} patents")
        logger.info(f"Index dimension: {dimension}, Index size: {self.index.ntotal}")

    def search(self, query, top_k=5):
        """Search patents using vector similarity"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("No patents indexed. Please index patents first.")
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
                    'patent_date': doc.get('patent_date', ''),
                    'score': float(score),
                    'content': doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                })
        
        return pd.DataFrame(results)

    def get_system_stats(self):
        """Get system statistics for monitoring"""
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stats = {
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100
        }
        
        return stats

    def test_search_queries(self):
        """Test the search system with various queries"""
        logger.info("=== Testing Patent Search System ===")
        
        test_cases = [
            {
                "query": "machine learning",
                "expected_keywords": ["machine learning", "algorithm", "neural"],
                "description": "Should find ML patents"
            },
            {
                "query": "artificial intelligence",
                "expected_keywords": ["artificial intelligence", "AI", "intelligent"],
                "description": "Should find AI patents"
            },
            {
                "query": "computer vision",
                "expected_keywords": ["computer vision", "image", "visual"],
                "description": "Should find computer vision patents"
            },
            {
                "query": "natural language processing",
                "expected_keywords": ["natural language", "text", "language"],
                "description": "Should find NLP patents"
            },
            {
                "query": "blockchain",
                "expected_keywords": ["blockchain", "distributed", "ledger"],
                "description": "Should find blockchain patents"
            },
            {
                "query": "internet of things",
                "expected_keywords": ["internet of things", "IoT", "connected"],
                "description": "Should find IoT patents"
            }
        ]
        
        total_tests = len(test_cases)
        passed_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n--- Test {i}/{total_tests}: {test_case['description']} ---")
            logger.info(f"Query: '{test_case['query']}'")
            
            results = self.search(test_case['query'], top_k=3)
            
            if not results.empty:
                logger.info("Top Results:")
                for _, row in results.iterrows():
                    logger.info(f"  {row['rank']}. {row['patent_id']} - {row['title']} (Score: {row['score']:.3f})")
                
                # Check if results contain expected keywords
                top_result_content = results.iloc[0]['content'].lower()
                expected_keywords = [kw.lower() for kw in test_case['expected_keywords']]
                found_keywords = [kw for kw in expected_keywords if kw in top_result_content]
                
                if found_keywords:
                    logger.info(f"✓ Found expected keywords: {found_keywords}")
                    passed_tests += 1
                else:
                    logger.warning(f"✗ Missing expected keywords. Found: {found_keywords}")
            else:
                logger.warning("No results found")
        
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Passed: {passed_tests}/{total_tests} tests ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! The search system is working correctly.")
        else:
            logger.warning("⚠️  Some tests failed. Review the results above.")

if __name__ == "__main__":
    # Example usage
    searcher = PatentSearch()
    
    # Display system stats
    try:
        stats = searcher.get_system_stats()
        logger.info(f"System Resources:")
        logger.info(f"  Memory: {stats['memory_available_gb']:.1f} GB available of {stats['memory_total_gb']:.1f} GB total")
        logger.info(f"  Disk: {stats['disk_free_gb']:.1f} GB available of {stats['disk_total_gb']:.1f} GB total")
    except ImportError:
        logger.info("psutil not available - skipping system stats")
    
    logger.info("Initializing Patent Search System...")
    
    # Fetch real patent data from local file
    patents = searcher.fetch_patents()
    
    if patents:
        logger.info(f"Loaded {len(patents)} patents")
        
        # Index the patents
        documents = searcher.process_patents(patents)
        searcher.index_patents(documents)
        
        # Run automated tests
        searcher.test_search_queries()
        
        # Example search
        logger.info("\n=== Example Search ===")
        results = searcher.search("machine learning healthcare", top_k=5)
        if not results.empty:
            logger.info("Top 5 results for 'machine learning healthcare':")
            for _, row in results.iterrows():
                logger.info(f"  {row['rank']}. {row['patent_id']} - {row['title']}")
    else:
        logger.error("No patents loaded. Check your data source.")
