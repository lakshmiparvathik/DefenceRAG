# DefenceRAG: Document-Scoped Retrieval for Grounded Policy QA

## Overview

This project was developed for the **DefenceRAG: Procurement & Policy Reasoning Challenge** hosted on Kaggle as part of the **AI Multi-Domain Hackathon by INICAI**.

The system answers questions about defence procurement policies by retrieving relevant information from official documents instead of generating unsupported responses.

## Features

- Document-scoped retrieval
- Two-stage TF-IDF ranking
- Sentence-level answer extraction
- Grounded answers from official policy documents
- Automatic source and section prediction

## Approach

The retrieval pipeline consists of the following steps:

1. Load the provided metadata and test dataset.
2. Detect the target document referenced in each question.
3. Restrict retrieval to the detected document.
4. Apply TF-IDF to retrieve the most relevant document chunks.
5. Perform sentence-level TF-IDF ranking within the top retrieved chunks.
6. Return the highest-scoring sentence together with its source document and section.

This document-scoped retrieval strategy reduces cross-document confusion and improves answer grounding.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Regular Expressions (Regex)

## Dataset

The project uses the datasets provided in the DefenceRAG Kaggle competition, including:

- Defence Procurement Manual (DPM 2025 Volume I & II)
- Delegation of Financial Powers for Defence Services (DFPDS 2024)
- Navy Regulations (Parts I–IV)

## Output

The generated submission file contains:

- id
- prediction
- pred_source
- pred_section

## Future Improvements

- Dense embedding retrieval
- Cross-encoder re-ranking
- Hybrid BM25 + semantic search
- LLM-based answer generation with retrieved context
- Page-level and clause-level citation support

## Author

**Sri Naga Lakshmi Parvathi Kurasala**

Developed for the **AI Multi-Domain Hackathon by INICAI (2026)**.
