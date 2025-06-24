#!/usr/bin/env python3
"""
Demonstration script for the patent search system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patent_search import PatentSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_search():
    """Demonstrate the patent search system with various queries"""
    
    # Initialize the search system
    logger.info("Initializing Patent Search System...")
    searcher = PatentSearch()
    
    # Load a small dataset for demonstration
    logger.info("Loading patent data...")
    patents = searcher.load_local_patent_data(
        max_patents=5000,  # 5K patents for demo
        filter_keywords=['artificial intelligence', 'machine learning', 'deep learning'],
        sample_fraction=0.2
    )
    
    if not patents:
        logger.error("No patents loaded")
        return
    
    logger.info(f"Loaded {len(patents)} patents")
    
    # Process and index patents
    documents = searcher.process_patents(patents)
    searcher.index_patents(documents)
    
    # Demo searches
    demo_queries = [
        "machine learning healthcare",
        "artificial intelligence autonomous vehicles", 
        "computer vision medical imaging",
        "natural language processing chatbots",
        "blockchain supply chain",
        "cybersecurity machine learning",
        "IoT smart home",
        "deep learning drug discovery"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("PATENT SEARCH DEMONSTRATION")
    logger.info("="*60)
    
    for i, query in enumerate(demo_queries, 1):
        logger.info(f"\n--- Search {i}: '{query}' ---")
        
        results = searcher.search(query, top_k=3)
        
        if not results.empty:
            logger.info("Top 3 Results:")
            for _, row in results.iterrows():
                logger.info(f"  {row['rank']}. Patent {row['patent_id']} (Score: {row['score']:.3f})")
                logger.info(f"     Content: {row['content'][:150]}...")
                logger.info("")
        else:
            logger.info("No results found")
    
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    demo_search() 