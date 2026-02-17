# Vector Databases Explained

## What Are Vector Databases?

Vector databases are specialized database systems designed to store, index, and query high-dimensional vector embeddings efficiently. Unlike traditional databases that work with structured data (rows and columns), vector databases excel at similarity search — finding items that are "close" to a query in vector space.

## How Do They Work?

### 1. Embedding Generation
Text, images, or other data are converted into numerical vectors (embeddings) using machine learning models. For example, the sentence "The cat sat on the mat" might become a vector of 1,536 floating-point numbers using OpenAI's text-embedding-3-small model.

### 2. Indexing
Vector databases use specialized indexing algorithms to organize vectors for fast retrieval:

- **HNSW (Hierarchical Navigable Small World)**: Creates a multi-layer graph structure. Used by ChromaDB, Qdrant, and Weaviate. Offers excellent query speed with good recall.
- **IVF (Inverted File Index)**: Clusters vectors and searches only relevant clusters. Used by FAISS.
- **LSH (Locality-Sensitive Hashing)**: Hashes similar vectors to the same buckets. Fast but lower recall.
- **Product Quantization (PQ)**: Compresses vectors for memory efficiency. Often combined with IVF.

### 3. Similarity Search
When a query comes in, it's converted to a vector and compared against indexed vectors using distance metrics:

- **Cosine Similarity**: Measures the angle between vectors. Most common for text.
- **Euclidean Distance (L2)**: Measures straight-line distance. Good for spatial data.
- **Dot Product**: Efficient for normalized vectors. Often used in recommendation systems.

## Popular Vector Databases

### ChromaDB
- **Type**: Embedded (runs in-process) or client-server
- **Best for**: Prototyping, small-to-medium datasets, local development
- **Key features**: Simple API, Python-native, automatic persistence
- **Indexing**: HNSW
- **License**: Apache 2.0

### FAISS (Facebook AI Similarity Search)
- **Type**: Library (not a database)
- **Best for**: Large-scale similarity search, research
- **Key features**: GPU support, multiple index types, highly optimized
- **Indexing**: IVF, PQ, HNSW, and combinations
- **License**: MIT

### Pinecone
- **Type**: Managed cloud service
- **Best for**: Production deployments, teams wanting zero ops
- **Key features**: Fully managed, auto-scaling, hybrid search
- **Pricing**: Free tier available, then usage-based

### Weaviate
- **Type**: Self-hosted or cloud
- **Best for**: Multi-modal search, GraphQL fans
- **Key features**: Built-in vectorization, hybrid search, multi-tenancy

### Qdrant
- **Type**: Self-hosted or cloud
- **Best for**: Production workloads, filtering + vector search
- **Key features**: Rich filtering, payload indexing, Rust-based performance

## Use Cases

1. **Semantic Search**: Finding documents by meaning, not just keywords
2. **Recommendation Systems**: "Users who liked X also liked Y"
3. **RAG (Retrieval-Augmented Generation)**: Grounding LLM responses in factual data
4. **Image Similarity**: Finding visually similar products or images
5. **Anomaly Detection**: Identifying outliers in high-dimensional data
6. **Question Answering**: Finding relevant passages to answer user questions

## Choosing the Right Vector Database

Consider these factors:
- **Scale**: How many vectors? Thousands → ChromaDB. Millions → FAISS or Qdrant.
- **Deployment**: Need managed? → Pinecone. Self-hosted? → Qdrant, Weaviate.
- **Integration**: What frameworks are you using? LangChain and LlamaIndex support most options.
- **Filtering**: Need metadata filtering alongside vector search? → Qdrant, Weaviate, Pinecone.
- **Cost**: Budget-sensitive? → ChromaDB or FAISS (free, local).

## Best Practices

1. **Choose the right embedding model** for your domain
2. **Chunk documents appropriately** — too small loses context, too large dilutes relevance
3. **Use metadata filtering** to narrow search scope when possible
4. **Monitor recall and precision** — fast isn't useful if results are wrong
5. **Re-index periodically** as your embedding model or chunking strategy evolves
