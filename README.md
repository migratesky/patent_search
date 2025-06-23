# Patent Search System

A RAG-based patent search system using:
- USPTO API for patent data
- Sentence Transformers for embeddings
- FAISS for vector search
- Haystack for RAG pipeline

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get a free API key from [PatentsView](https://patentsview.org/) and set it as environment variable:
```bash
export PATENTSVIEW_API_KEY="your_api_key"
```

3. Run the system:
```bash
python patent_search.py
```

## Customization

Modify `query_params` in `patent_search.py` to:
- Change the patent search criteria
- Request different patent fields
- Adjust the number of patents to fetch

## Searching

The system will automatically index patents after fetching. You can then search using:
```python
results = searcher.search("your search query")
```
