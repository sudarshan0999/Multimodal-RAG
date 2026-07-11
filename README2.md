# Multimodal RAG - Detailed UI Parameter Guide

This guide explains all parameters available in the Streamlit UI for the Multimodal RAG application. Use this as a reference to configure the system for your specific use case.

---

## Table of Contents

1. [LLM Provider](#1-llm-provider)
2. [Chunking](#2-chunking)
3. [Retrieval](#3-retrieval)
4. [Embeddings](#4-embeddings)
5. [Reranking](#5-reranking)
6. [Generation](#6-generation)
7. [Quick Reference](#7-quick-reference)
8. [Common Scenarios](#8-common-scenarios)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. LLM Provider

### Provider Selection

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Provider | groq, openai, azure_openai, ollama | groq | LLM provider for chat and image captioning |

### Options Explained

#### Groq (Default)
- **Pros**: Fast inference, low cost, excellent for real-time applications
- **Cons**: Limited model selection
- **Required API Key**: `GROQ_API_KEY` in `.env`
- **Recommended Models**: 
  - Chat: `llama-3.3-70b-versatile`
  - Vision: `llama-3.2-11b-vision-preview`

#### OpenAI
- **Pros**: Access to GPT-4o, GPT-4o-mini, extensive model ecosystem
- **Cons**: Higher cost, rate limits
- **Required API Key**: `OPENAI_API_KEY` in `.env`
- **Recommended Models**:
  - Chat: `gpt-4o-mini` (fast) or `gpt-4o` (powerful)
  - Vision: `gpt-4o-mini` or `gpt-4o`

#### Azure OpenAI
- **Pros**: Enterprise-grade security, custom deployments
- **Cons**: Requires Azure subscription, more complex setup
- **Required Settings**:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT` (e.g., `https://your-resource.openai.azure.com/`)
  - `AZURE_OPENAI_API_VERSION` (e.g., `2024-02-15-preview`)
  - Chat deployment name
  - Vision deployment name

#### Ollama (Local)
- **Pros**: Runs locally, no API costs, privacy-friendly
- **Cons**: Requires local GPU/CPU resources, slower for large models
- **Required Settings**:
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- **Recommended Models**:
  - Chat: `llama3.2`
  - Vision: `llava`

### API Key Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| Load API keys from .env | Checkbox | If enabled, reads API key from `.env` file |
| API Key | Text Input | Direct API key input (overrides `.env` if provided) |

**Note**: Always prioritize security - use `.env` files for production deployments.

---

## 2. Chunking

### Chunking Strategy

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Strategy | fixed, recursive, token, semantic | fixed | Text chunking algorithm |
| Chunk size | 64-8000 | 1000 | Maximum characters/tokens per chunk |
| Overlap | 0-2000 | 200 | Overlap between adjacent chunks |

### Strategy Details

#### Fixed Chunking
- **How it works**: Divides text into fixed-size chunks without considering structure
- **Best for**: Simple documents, fast processing, when context doesn't matter
- **Pros**: Fast, predictable output size
- **Cons**: May split mid-sentence, loses context boundaries

```
Example: text[0:1000] → chunk1
         text[800:1800] → chunk2 (800 overlap)
         text[1600:2600] → chunk3
```

#### Recursive Chunking
- **How it works**: Hierarchical splitting - first by paragraphs, then sentences, then words
- **Best for**: Structured documents (papers, reports, books)
- **Pros**: Respects document structure, better context
- **Cons**: Slightly slower, variable chunk sizes

```
Hierarchy: Document → Paragraphs → Sentences → Words
Splits at: \n\n → \n → . → space
```

#### Token Chunking
- **How it works**: Uses tiktoken to count tokens, splits at token boundaries
- **Best for**: When precise token count matters (API limits)
- **Pros**: Exact token control, API-friendly
- **Cons**: Requires tiktoken installation, slightly slower

#### Semantic Chunking
- **How it works**: Embeds text sections, splits at embedding similarity breakpoints
- **Best for**: Documents requiring semantic coherence
- **Pros**: Chemically meaningful chunks
- **Cons**: Slower (requires embeddings for each chunk)

### Chunk Size Guidelines

| Document Type | Recommended Size | Reasoning |
|--------------|------------------|------------|
| Technical docs | 800-1200 | Complex concepts need context |
| Papers/Reports | 1000-1500 | Balance detail and context |
| Books | 1500-2000 | Large context windows |
| Short documents | 500-800 | Less content overall |

### Overlap Guidelines

| Scenario | Recommended Overlap | Reasoning |
|----------|-------------------|------------|
| High recall | 200-400 | More context continuity |
| Low latency | 50-100 | Smaller chunks, faster processing |
| Semantic chunks | 0 | Natural breakpoints only |

---

## 3. Retrieval

### Retrieval Parameters

| Parameter | Range | Default | Description |
|-----------|---------|---------|-------------|
| Top-k | 1-50 | 8 | Final number of results returned to LLM |
| Semantic pool (fetch_k) | 5-200 | 40 | Initial candidate pool for semantic search |
| Extra images per query | 0-12 | 4 | Additional images to retrieve |
| Hybrid semantic weight | 0.0-1.0 | 0.6 | Weight for semantic search |
| Hybrid BM25 weight | 0.0-1.0 | 0.4 | Weight for BM25 keyword search |

### Parameter Explanations

#### Top-k
- **What it does**: Number of documents passed to the LLM as context
- **Impact**: More context = richer answers but higher latency/cost
- **Guidelines**:
  - 4-6: Simple questions
  - 8-12: General use (recommended)
  - 12-20: Complex analytical questions

#### Semantic Pool (fetch_k)
- **What it does**: Initial number of candidates retrieved via semantic search
- **Impact**: Larger pool = more diverse candidates but more processing
- **Guidelines**:
  - Should be 3-5x Top-k for best results
  - 20-40 works well for most cases
  - Increase for diverse document collections

#### Extra Images per Query
- **What it does**: Ensures diagrams/charts appear in results regardless of text match
- **Impact**: Higher values = more visual context but may dilute text relevance
- **Guidelines**:
  - 0: No extra images
  - 2-4: Standard (recommended)
  - 6-8: Image-heavy documents (papers, manuals)

#### Hybrid Weights

The hybrid search combines semantic (embedding-based) and BM25 (keyword-based) search using Reciprocal Rank Fusion (RRF):

```
Score = (semantic_weight / (k + semantic_rank)) + (bm25_weight / (k + bm25_rank))
where k = 60 (default rrf_k)
```

| Weight Combination | Best For |
|-------------------|----------|
| sem=0.6, bm25=0.4 | General use (default) |
| sem=0.7-0.8 | Conceptual questions |
| sem=0.5, bm25=0.5 | Balanced |
| sem=0.3-0.4, bm25=0.6-0.7 | Keyword-heavy queries |

**Key Rule**: semantic weight + BM25 weight should approximately equal 1.0

---

## 4. Embeddings

### Embedding Backend

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Embed backend | sentence_transformers, openai, azure_openai | sentence_transformers | Embedding model backend |
| Embed model / deployment | Text | sentence-transformers/all-MiniLM-L6-v2 | Model name |

### Backend Details

#### Sentence Transformers (Default)
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Pros**: Free, runs locally, fast, no API costs
- **Cons**: Lower accuracy than commercial models
- **Use case**: Groq users, cost-sensitive applications

#### OpenAI
- **Model**: text-embedding-3-small
- **Dimensions**: 1536
- **Pros**: Higher accuracy, managed service
- **Cons**: Requires API key, per-request costs
- **Use case**: Teams with OpenAI subscriptions

#### Azure OpenAI
- **Model**: Custom deployment
- **Dimensions**: Configurable
- **Pros**: Enterprise security, custom fine-tuned models
- **Cons**: Requires Azure setup
- **Use case**: Enterprise deployments

### Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|-------|------------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Free |
| text-embedding-3-small | 1536 | Fast | Better | Per request |
| text-embedding-3-large | 3076 | Medium | Best | Per request |

---

## 5. Reranking

### Reranker Selection

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Reranker | none, ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-v2-m3 | ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking model |

### Reranker Details

#### How Reranking Works

1. Initial retrieval gets Top-k candidates
2. Reranker scores each (query, document) pair using a cross-encoder
3. Results are re-sorted by cross-encoder scores
4. Final Top-k returned to LLM

```
Cross-Encoder: (query, document) → score
Example: ("what is neural network?", "Neural networks are...") → 0.95
         ("what is neural network?", "Network architecture...") → 0.72
```

#### Options Explained

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **none** | - | Instant | - | Testing, fast prototyping |
| **ms-marco-MiniLM-L-6-v2** | ~100MB | Fast | Medium | Most cases (default) |
| **BAAI/bge-reranker-v2-m3** | ~500MB | Slow | High | Accuracy-critical |

#### MS MARCO MiniLM L-6-v2
- **Training Data**: MS MARCO (Microsoft Machine Reading Comprehension)
- **Parameters**: ~100 million
- **Speed**: ~100 docs/sec on CPU
- **Accuracy**: Good for general queries

#### BGE Reranker v2-m3
- **Training Data**: BAAI's large-scale training
- **Parameters**: ~500 million
- **Speed**: ~20 docs/sec on CPU
- **Accuracy**: Excellent, especially for complex queries

**Installation Note**: BGE reranker will download ~500MB on first use.

---

## 6. Generation

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Stream responses | Checkbox | On | Real-time word-by-word output |
| Force re-index on next upload | Checkbox | Off | Reprocess existing PDFs |

### Stream Responses

- **On (Recommended)**: Displays LLM response as it's generated
- **Pros**: Better UX, feels more responsive, faster perceived response
- **Cons**: Slightly higher resource usage

- **Off**: Waits for complete response before displaying
- **Pros**: Complete answer available
- **Cons**: Slower perceived response

### Force Re-index

- **Off (Default)**: Skips PDFs already in database (identified by hash)
- **On**: Processes PDF again, overwrites existing entries

**Use when**:
- Chunking strategy changed
- Embedding model changed
- PDF was corrupted during initial processing
- Debugging ingestion issues

---

## 7. Quick Reference

### Standard Configurations

| Use Case | Provider | Chunking | Top-k | Reranker | Stream |
|----------|----------|----------|------|----------|--------|
| Fast testing | groq | fixed, 1000 | 8 | none | On |
| General use | groq | recursive, 1000 | 10 | MiniLM | On |
| High accuracy | openai | semantic, 800 | 12 | BGE | On |
| Charts-heavy | groq | recursive, 1200 | 10 | MiniLM | On |
| Large docs | azure | fixed, 2000 | 6 | none | On |

### Parameter Presets

#### Fast Prototype
```
Provider: groq
Strategy: fixed
Chunk size: 1000
Top-k: 8
Fetch_k: 20
Reranker: none
Stream: on
```

#### Balanced
```
Provider: groq/openai
Strategy: recursive
Chunk size: 1000
Top-k: 10
Fetch_k: 40
Image_top_k: 4
Sem_weight: 0.6
BM25_weight: 0.4
Reranker: MiniLM
Stream: on
```

#### Maximum Accuracy
```
Provider: openai (gpt-4o)
Strategy: semantic
Chunk size: 800
Top-k: 12
Fetch_k: 60
Image_top_k: 6
Sem_weight: 0.5
BM25_weight: 0.5
Reranker: BGE
Stream: on
```

#### Image/Paper Heavy
```
Provider: groq
Strategy: recursive
Chunk size: 1200
Top-k: 10
Fetch_k: 50
Image_top_k: 8
Sem_weight: 0.5
BM25_weight: 0.5
Reranker: MiniLM
Stream: on
```

#### Production
```
Provider: openai/azure
Strategy: recursive
Chunk size: 1000
Top-k: 8
Fetch_k: 40
Reranker: BGE
Stream: on
Force_reindex: Off
```

---

## 8. Common Scenarios

### Scenario 1: Research Paper Q&A
```
- Strategy: recursive, size=1200, overlap=200
- Top-k: 10, fetch_k: 50
- Image_top_k: 8 (papers have many figures)
- Sem_weight: 0.5, BM25_weight: 0.5
- Reranker: MiniLM or BGE (for accuracy)
- Show tables: Always
```

### Scenario 2: Technical Manual/Documentation
```
- Strategy: recursive, size=800, overlap=150
- Top-k: 8, fetch_k: 40
- Image_top_k: 6
- Sem_weight: 0.6, BM25_weight: 0.4
- Reranker: MiniLM
```

### Scenario 3: Financial Reports
```
- Strategy: semantic, size=1000
- Top-k: 12, fetch_k: 60 (tables important)
- Image_top_k: 4
- Sem_weight: 0.5, BM25_weight: 0.5
- Reranker: BGE (numbers accuracy)
```

### Scenario 4: Books/E-books
```
- Strategy: fixed, size=2000, overlap=400
- Top-k: 6, fetch_k: 30
- Image_top_k: 0
- Sem_weight: 0.7, BM25_weight: 0.3
- Reranker: none (context in chunks)
```

### Scenario 5: Quick Exploration
```
- Strategy: fixed, size=1000
- Top-k: 6, fetch_k: 20
- Image_top_k: 2
- Sem_weight: 0.6, BM25_weight: 0.4
- Reranker: none
- Stream: on
```

---

## 9. Troubleshooting

### Performance Issues

| Problem | Likely Cause | Solution |
|---------|------------|----------|
| Slow initial load | Model download | Wait for first-time download |
| Slow retrieval | Large fetch_k | Reduce to 20-30 |
| Slow response | Large chunks | Reduce chunk size |
| Slow reranking | BGE model | Use MiniLM or none |

### Quality Issues

| Problem | Likely Cause | Solution |
|---------|------------|----------|
| Missing results | fetch_k too small | Increase fetch_k |
| Missing images | image_top_k=0 | Increase image_top_k |
| Irrelevant results | Sem weight high | Increase BM25 weight |
| Wrong answer | No reranking | Enable reranking |

### Technical Issues

| Problem | Solution |
|---------|----------|
| API key error | Check .env file |
| Embedding timeout | Use sentence_transformers |
| Table extraction fail | Check Camelot installation |
| Import errors | Run from project root |

---

## Environment Variables Reference

```bash
# LLM Providers
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_VISION_DEPLOYMENT=your_vision_deployment

# Model Configurations
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini

# Embeddings
EMBED_BACKEND=sentence_transformers
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Storage Paths
CHROMA_PERSIST_DIR=./chroma_data
UPLOAD_DIR=./uploads
CAPTIONS_DIR=./captions
```

---

## Additional Resources

- **Architecture Diagram**: See main README.md
- **API Documentation**: OpenAI, Groq, Azure docs
- **Model Cards**: Hugging Face model pages

---

*Last Updated: April 2025*