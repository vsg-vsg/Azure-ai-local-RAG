## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline that combines information retrieval with large language models (LLMs) to answer user queries. It supports two modes:

Local Mode – Uses Sentence Transformers (all-MiniLM-L6-v2) to embed documents and retrieve the most relevant passages, paired with a Hugging Face text generation model (TinyLlama-1.1B-Chat) for response generation.

Azure Mode – Uses Azure AI Search for document retrieval and Azure OpenAI Service (e.g., GPT-4 deployment) for answer generation.

## Technologies Used

- Sentence Transformers (SBERT) – Efficient embeddings for semantic similarity search between queries and documents.

- Hugging Face Transformers – Provides lightweight, open-source LLMs (TinyLlama in this case) for text generation.

- Azure AI Search – Enterprise-grade vector & keyword search engine for document retrieval.

- Azure OpenAI Service – Hosted LLMs (GPT-4, GPT-35) for scalable, high-quality natural language generation.

- dotenv – Securely handles API keys and configuration via environment variables.

## Pipeline Flow

### Retrieval

- Local Mode: Encode query + documents with SBERT embeddings, rank with cosine similarity.

- Azure Mode: Query Azure AI Search to fetch the most relevant documents.

### Generation

- Local Mode: Feed retrieved context into a Hugging Face model for answer generation.

- Azure Mode: Construct chat-style prompts for Azure OpenAI (GPT models) to generate context-aware responses.

### End-to-End RAG

User query → Retrieve relevant docs → LLM generates answer → Return to user.
