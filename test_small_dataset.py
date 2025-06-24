#!/usr/bin/env python3
"""
Test script for the patent search system with a small dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patent_search import PatentSearch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_small_dataset():
    """Test with a small dataset to verify functionality"""
    searcher = PatentSearch()
    
    logger.info("=== Testing Small Dataset ===")
    
    # Test with just 2023 data and limited patents
    try:
        # Download only 2023 data
        logger.info("Downloading 2023 patent data...")
        downloaded_files = searcher.download_patent_data(year_start=2023, year_end=2023)
        
        if not downloaded_files:
            logger.error("No files downloaded")
            return False
        
        # Load a small subset (10K patents) for testing
        logger.info("Loading small subset of patents...")
        patents = searcher.load_patent_data(
            year_start=2023,
            year_end=2023,
            max_patents=10000,  # Only 10K patents for testing
            filter_keywords=['artificial intelligence', 'machine learning', 'deep learning'],
            sample_fraction=0.5  # Use 50% of available data
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
            return True
        else:
            logger.error("Search returned no results")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_small_dataset()
    if success:
        logger.info("✅ Small dataset test passed!")
    else:
        logger.error("❌ Small dataset test failed!")
        sys.exit(1) 