"""Simulated in-memory knowledge base for RAG-style retrieval demos."""

SIMULATED_KB: dict[str, str] = {
    "python": "Python is a high-level, interpreted programming language known for simplicity and readability.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph-based workflows.",
    "openai": "OpenAI develops AI systems including GPT-4o, a multimodal model supporting text, image, and audio.",
    "rag": "RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs to ground responses in external knowledge.",
    "agent": "An AI agent autonomously uses tools and reasoning to complete tasks across multiple steps.",
    "semantic search": "Semantic search finds results based on meaning/intent rather than exact keyword matching.",
    "vector database": (
        "A vector database stores high-dimensional embeddings for fast similarity search. "
        "Well-suited for semantic search, RAG pipelines, recommendation systems, and ML feature stores. "
        "Popular options: Pinecone, Weaviate, Qdrant, pgvector. "
        "Strengths: fast ANN search, scales to millions of vectors, integrates well with LLM stacks. "
        "Weaknesses: not ideal for structured/relational queries, overkill for small datasets."
    ),
    "vectordb vs relational": (
        "Vector databases excel at unstructured similarity search; relational databases excel at structured, "
        "transactional workloads. For hybrid needs, pgvector adds vector search on top of PostgreSQL."
    ),
    "vectordb use cases": (
        "Best use cases for vector databases: semantic document search, RAG, image similarity, "
        "product recommendations, anomaly detection, and chatbot memory."
    ),
    "mongodb": (
        "MongoDB is a NoSQL document database storing data as flexible JSON-like documents. "
        "Strengths: flexible schema, horizontal scaling via sharding, rich query language, large ecosystem. "
        "Weaknesses: higher memory usage, no native joins, not ideal for highly relational data."
    ),
    "mongodb vs vectordb": (
        "MongoDB Atlas now supports vector search, blurring the line between document and vector stores. "
        "Pure vector DBs (Pinecone, Qdrant) are faster for ANN-only workloads; MongoDB suits hybrid "
        "structured+vector needs within a single platform."
    ),
    "database selection": (
        "Choosing a database depends on: data structure, query patterns, scale, consistency requirements, "
        "and existing stack. There is no universally best database — the right choice is use-case driven."
    ),
    "database optimization": (
        "Optimization strategies vary by workload: "
        "(1) Query optimization: add indexes, rewrite slow queries. "
        "(2) Schema: normalize for write-heavy, denormalize for read-heavy. "
        "(3) Scaling: vertical for single-node, horizontal sharding for distributed. "
        "(4) Caching: Redis or Memcached. "
        "(5) Vector DB: tune HNSW parameters (ef, M) for ANN recall vs speed tradeoff."
    ),
    "smoking regulation june 2024": (
        "[Regulation: Smoking Policy — June 2024] "
        "Effective June 1, 2024: Smoking prohibited in all enclosed public spaces. "
        "Outdoor smoking areas permitted at least 10 meters from building entrances. "
        "Violations: $200 fine. E-cigarettes treated same as traditional cigarettes."
    ),
    "smoking regulation june 2025": (
        "[Regulation: Smoking Policy — June 2025] "
        "Effective June 1, 2025: Smoking prohibited within 5 meters of any building entrance. "
        "Designated smoking areas must be fully covered structures. "
        "Violations: $500 fine. Smoking near schools/hospitals: $1000 fine."
    ),
    "dayumtrade team": (
        "Dayumtrade is a small design consultancy founded in 2018. "
        "Team: CEO - James Park, Head Designer - Priya Nair, "
        "Lead Engineer - Sam Wong, Marketing Manager - Linda Chen."
    ),
    "priya nair": (
        "Priya Nair — Senior Designer, 10 years experience in product and interaction design. "
        "Education: BFA, Rhode Island School of Design (RISD), 2013. "
        "Master's in Interaction Design, Royal College of Art, London, 2015. "
        "Previously at IDEO London and Figma. Her father is Rajesh Nair."
    ),
    "rajesh nair": (
        "Rajesh Nair — name: Rajesh Nair. Senior COBOL Developer, 30 years experience in mainframe banking. "
        "Current location: Moscow, Russia. Works for Sberbank's legacy infrastructure team. "
        "Originally from Chennai, India. Father of Priya Nair, based in the UK."
    ),
}
