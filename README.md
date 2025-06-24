# Patent Search System

A RAG-based patent search system using real patent data with:
- Local patent description data (USPTO 2024)
- Sentence Transformers for embeddings
- FAISS for vector search
- Memory-efficient processing for large datasets

## ✅ Implementation Status

**Option 1: Moderate Dataset (Recommended)** - **COMPLETED**

- **Data Source**: Local USPTO patent description file (`g_detail_desc_text_2024.tsv.zip`)
- **Dataset Size**: 5K-100K patents (configurable)
- **Memory Usage**: ~2-8 GB RAM
- **Storage**: ~4.3 GB compressed data file
- **Performance**: ✅ Successfully tested with 5K patents

## Features

- **Real Patent Data**: Uses actual USPTO patent descriptions from 2024
- **Memory Efficient**: Processes data in chunks to manage memory usage
- **Smart Filtering**: Focuses on technology-relevant patents (AI, ML, etc.)
- **Vector Search**: Fast semantic search using FAISS
- **Progress Tracking**: Real-time progress bars and logging
- **Configurable**: Adjustable dataset size and filtering criteria

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the patent data file:
   - `g_detail_desc_text_2024.tsv.zip` (4.3 GB)
   - Contains patent descriptions from USPTO 2024

3. Run the system:
```bash
python patent_search.py
```

## Usage Examples

### Test with Small Dataset (5K patents)
```bash
python test_memory_efficient.py
```

### Test with Local Data (500 patents)
```bash
python test_local_data.py
```

### Full System with Moderate Dataset
```bash
python patent_search.py
```

## Configuration

Modify parameters in `patent_search.py`:

```python
# Dataset size
max_patents = 500000  # Number of patents to load

# Technology filtering keywords
filter_keywords = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'computer vision', 'natural language processing', 'blockchain',
    # ... add more keywords
]

# Sampling fraction (for memory efficiency)
sample_fraction = 0.3  # Use 30% of available data
```

## System Requirements

- **RAM**: 8-16 GB recommended
- **Storage**: 10-20 GB free space
- **CPU**: 4+ cores recommended
- **GPU**: Optional (MPS/CPU fallback available)

## Performance

- **5K patents**: ~2 minutes processing time
- **50K patents**: ~15-20 minutes processing time
- **100K patents**: ~30-45 minutes processing time
- **Search speed**: <1 second per query

## Data Structure

The system processes patent data with the following structure:
- **patent_id**: Unique patent identifier
- **description_text**: Patent description content
- **description_length**: Length of description

## Search Capabilities

The system can search for:
- Machine learning and AI patents
- Computer vision and image processing
- Natural language processing
- Blockchain and distributed systems
- Cybersecurity and cloud computing
- IoT and connected devices
- And more technology areas

## Example Searches

```python
# Initialize search system
searcher = PatentSearch()

# Search for specific technologies
results = searcher.search("machine learning healthcare")
results = searcher.search("artificial intelligence drug discovery")
results = searcher.search("blockchain medical records")
results = searcher.search("computer vision autonomous vehicles")
```

## Memory Management

The system includes several memory optimization features:
- **Chunked Processing**: Loads data in 10K patent chunks
- **Batch Embeddings**: Generates embeddings in 1K document batches
- **Text Truncation**: Limits patent descriptions to 2K characters
- **Smart Filtering**: Reduces dataset size through keyword filtering
- **Sampling**: Uses configurable sampling for large datasets

## Future Enhancements

- Add patent titles and abstracts (if available)
- Implement patent classification filtering
- Add date-based filtering
- Support for multiple years of data
- Web interface for search queries
- Export search results to various formats
