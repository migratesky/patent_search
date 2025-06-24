#!/usr/bin/env python3
"""
Test script for the patent search system with local patent data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patent_search import PatentSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_local_data():
    """Test with local patent data to verify functionality"""
    searcher = PatentSearch()
    
    logger.info("=== Testing Local Patent Data ===")
    
    try:
        # Load a small subset (5K patents) for testing
        logger.info("Loading small subset of patents from local file...")
        patents = searcher.load_local_patent_data(
            max_patents=5000,  # Only 5K patents for testing
            filter_keywords=['artificial intelligence', 'machine learning', 'deep learning'],
            sample_fraction=0.1  # Use 10% of available data
        )
        
        if not patents:
            logger.error("No patents loaded")
            return False
        
        logger.info(f"Successfully loaded {len(patents)} patents")
        
        # Process patents
        documents = searcher.process_patents(patents)
        logger.info(f"Processed {len(documents)} documents")
        
        if not documents:
            logger.error("No documents processed")
            return False
        
        # Index patents
        searcher.index_patents(documents)
        
        # Test search
        logger.info("Testing search functionality...")
        results = searcher.search("machine learning", top_k=3)
        
        if not results.empty:
            logger.info("Search test successful!")
            logger.info("Top results:")
            for _, row in results.iterrows():
                logger.info(f"  {row['rank']}. {row['patent_id']} - {row['title']}")
                logger.info(f"     Content preview: {row['content'][:100]}...")
            return True
        else:
            logger.error("Search returned no results")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_data()
    if success:
        logger.info("✅ Local data test passed!")
    else:
        logger.error("❌ Local data test failed!")
        sys.exit(1) 