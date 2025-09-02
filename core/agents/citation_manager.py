from typing import List, Dict, Tuple
import re
import hashlib
from core.models.state import Source, CitedContent


class CitationManager:
    def __init__(self):
        self.reset()  # Initialize clean

    # useful if you reuse instances
    def reset(self):
        """Reset the manager for a new query"""
        self.source_registry: Dict[str, Source] = {}
        self.citation_counter = 1
        self.url_to_id: Dict[str, str] = {}


    def add_source(self, source: Source) -> str:
        """Add a source and return its citation ID, reusing if URL exists"""
        for existing_id, existing in self.source_registry.items():
            if existing.url == source.url:
                return existing_id  # Reuse existing ID for duplicate URL
        source_id = f"src_{self.citation_counter}"
        self.source_registry[source_id] = source
        self.citation_counter += 1
        return source_id

    def create_cited_content(self, content: str, source_ids: List[str]) -> CitedContent:
        """Create content with embedded citations"""
        cited_content = content

        # Add citation markers
        for i, source_id in enumerate(source_ids):
            citation_marker = f"[{self.get_citation_number(source_id)}]"
            # Insert citation at end of relevant sentences
            cited_content = self.insert_citations(cited_content, citation_marker)

        return CitedContent(
            content=cited_content,
            source_ids=source_ids,
            confidence=self.calculate_confidence(source_ids)
        )

    def get_citation_number(self, source_id: str) -> int:
        """Get the display number for a citation"""
        source_ids = list(self.source_registry.keys())
        return source_ids.index(source_id) + 1 if source_id in source_ids else 0

    def insert_citations(self, text: str, citation: str) -> str:
        """Insert citations at appropriate positions in text"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences:
            sentences[-1] += citation
        return ' '.join(sentences)

    def format_sources_list(self) -> str:
        """Format sources for display, capped at 10 unique, sorted by ID"""

        def get_id_number(item):
            try:
                # item is a tuple (source_id, source); access source_id with item[0]
                id_str = item[0]  # e.g., 'src_1'
                number_part = id_str.split('_')[1]  # e.g., '1'
                return int(number_part)  # e.g., 1
            except (IndexError, ValueError, AttributeError):
                return float('inf')  # Push invalid items to the end (safe fallback)

        sorted_sources = sorted(self.source_registry.items(), key=get_id_number)
        sources_text = "\n\n**Sources:**\n"
        seen_urls = set()  # Extra dedup for safety
        count = 0
        for source_id, source in sorted_sources:
            if source.url not in seen_urls and count < 10:
                count += 1
                sources_text += f"{count}. [{source.title}]({source.url}) - {source.domain}\n"
                seen_urls.add(source.url)
        return sources_text

    def calculate_confidence(self, source_ids: List[str]) -> float:
        """Calculate confidence based on source quality and quantity"""
        if not source_ids:
            return 0.0

        total_relevance = sum(
            self.source_registry[sid].relevance_score
            for sid in source_ids if sid in self.source_registry
        )
        return min(total_relevance / len(source_ids), 1.0)
