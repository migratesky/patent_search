#!/usr/bin/env python3
"""
Test script for memory-efficient patent search with local data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patent_search import PatentSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_efficient():
    """Test with a moderate dataset using memory-efficient approach"""
    searcher = PatentSearch()
    
    logger.info("=== Testing Memory-Efficient Patent Search ===")
    
    try:
        # Load a moderate subset (50K patents) for testing
        logger.info("Loading moderate subset of patents from local file...")
        patents = searcher.load_local_patent_data(
            max_patents=5000,  # 50K patents for testing
            filter_keywords=['artificial intelligence', 'machine learning', 'deep learning'],
            sample_fraction=0.15  # Use 15% of available data
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
        
        # Index patents with smaller batch size
        logger.info("Indexing patents with memory-efficient approach...")
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
    success = test_memory_efficient()
    if success:
        logger.info("✅ Memory-efficient test passed!")
    else:
        logger.error("❌ Memory-efficient test failed!")
        sys.exit(1) 