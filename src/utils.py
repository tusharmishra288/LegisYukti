"""
Utility functions for legal text processing and context management.

This module provides specialized utilities for:
- Context pruning with legal citation preservation
- Feedback text cleaning and normalization
- Query parsing with domain-aware sanitization
"""

import re
from loguru import logger
from langchain_core.output_parsers import BaseOutputParser

def prune_legal_context(context, max_chars: int = 3500) -> str:
    """Safely prune legal context while preserving citation integrity.

    Intelligently truncates retrieved legal context to fit token limits while ensuring
    that legal citations and breadcrumbs remain complete. Prevents cutting off
    important references like "passage: [LAW: Indian Contract Act | Section 73]".

    Args:
        context: Retrieved legal context (string or list of strings)
        max_chars: Maximum character limit for the pruned context

    Returns:
        str: Pruned context that fits within limits without breaking citations
    """
    # Handle different input formats from retrieval system
    if isinstance(context, list):
        context = "\n\n".join([str(c) for c in context])

    if not isinstance(context, str):
        context = str(context)

    # Aggressive whitespace cleanup to maximize content within token limits
    context = re.sub(r'\s+', ' ', context).strip()

    if len(context) <= max_chars:
        return context

    # Smart pruning: Find the last complete legal citation tag
    pruned = context[:max_chars]
    last_tag_index = pruned.rfind('passage: [LAW:')

    # Cut before incomplete citations to maintain reference integrity
    if last_tag_index > 0:
        return pruned[:last_tag_index].strip()

    return pruned.strip()

def clean_feedback(text: str) -> str:
    """Clean and normalize auditor feedback text for user display.

    Removes technical prefixes and ensures feedback ends at natural sentence boundaries.
    Prevents truncated or awkwardly formatted audit feedback from being shown to users.

    Args:
        text: Raw feedback text from auditing system

    Returns:
        str: Clean, properly formatted feedback text
    """
    # Remove common technical prefixes from audit responses
    text = re.sub(r'^(The response|Evaluation|REASON|Feedback|Score)[:\s]*', '', text, flags=re.IGNORECASE)

    # Extract first 1-2 complete sentences for concise feedback
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = " ".join(sentences[:2])

    # Ensure proper sentence termination
    if not clean.endswith(('.', '!', '?')):
        last_punct = max(clean.rfind('.'), clean.rfind('!'), clean.rfind('?'))
        if last_punct > 0:
            clean = clean[:last_punct + 1]
        else:
            clean = clean[:147] + "..." # Fallback for very long single sentences

    return clean.strip()

class StrictLegalQueryParser(BaseOutputParser[list[str]]):
    """Domain-aware query parser that prevents cross-contamination between legal domains.

    Specialized parser for multi-query retrieval that:
    - Cleans formatting artifacts from LLM-generated queries
    - Prevents criminal law keywords from polluting civil/family searches
    - Ensures domain-appropriate search terms for accurate retrieval

    This prevents "zero results" scenarios where criminal-specific terms
    contaminate searches for civil matters in the vector store.
    """

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        raw_clean_queries = []

        # Phase 1: Standard cleaning of formatting artifacts
        for line in lines:
            line = re.sub(r"^(?:\d+\.?|[-*•]|Query \d+:|\*\*Query.*?\*\*|As a .*?:|Here are .*?:)\s*", "", line.strip())
            line = line.replace('"', '').replace("'", "").replace("+", " ").replace("*", "").strip()

            if len(line) > 10 and not line.endswith(":"):
                raw_clean_queries.append(line)

        # Phase 2: Domain sanitization to prevent cross-contamination
        final_queries = []
        civil_keywords = [
            "sale deed", "registration", "marriage", "divorce", "apartment", "flat",
            "notarized", "agreement", "tenant", "cpc", "civil", "it act", "cyber",
            "pan card", "whatsapp fraud", "inheritance", "succession", "will",
            "father", "son", "share", "constitution", "fundamental rights",
            "ndps", "drugs", "narcotics", "article"
        ]
        criminal_noise = [
            "under bns", "under bnss", "in bns", "in bnss", "ipc equivalent",
            "punishment and bailability", "corresponding bns", "bns section"
        ]

        for query in raw_clean_queries:
            # Detect civil/family/property matters and sanitize criminal keywords
            if any(key in query.lower() for key in civil_keywords):
                # Remove criminal-specific terms that cause retrieval failures
                for noise in criminal_noise:
                    query = re.sub(re.escape(noise), "", query, flags=re.IGNORECASE)

                # Clean up spacing artifacts from regex removal
                query = re.sub(r'\s+', ' ', query).strip()

            final_queries.append(query)

        # Deduplicate while preserving order
        unique_queries = list(dict.fromkeys(final_queries))

        output = unique_queries[:3]
        logger.info(f"🧹 MMR Cleaned & Sanitized Queries: {output}")
        return output