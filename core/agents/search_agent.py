from typing import List, Dict, Any
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from core.models.state import AgentState, Source, SearchResult
from core.agents.citation_manager import CitationManager
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

class SearchAgent:
    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b")
        self.tavily = TavilySearch(max_results=5, search_depth="advanced")
        self.citation_manager = CitationManager()
        self.citation_manager.reset()

    def search_and_analyze(self, state: AgentState):
        """Main search and analysis pipeline"""
        self.citation_manager.reset()  # Clear for this query
        query = state["current_query"]

        # 1. generate search queries
        search_queries = self.generate_search_queries(query)
        search_queries = [query] + search_queries
        print("Search queries:", search_queries)

        # 2. Execute searches
        all_sources = []
        search_results = []

        for search_query in search_queries:
            results = self.tavily.invoke(search_query)
            sources = self.process_search_results(results, search_query)
            all_sources.extend(sources)

            search_results.append(SearchResult(
                query=search_query,
                sources=sources,
                total_results=len(sources),
                search_time=datetime.now()
            ))

        # 3. Deduplicate and rank sources
        unique_sources = self.deduplicate_sources(all_sources)
        ranked_sources = self.rank_sources(unique_sources, query)

        # 4. Add sources to citation manager
        source_registry = {}
        indexed_sources = {}

        for i, source in enumerate(ranked_sources[:10], start=1):
            source_id = self.citation_manager.add_source(source)
            source_registry[source_id] = source
            indexed_sources[i] = {"url": source.url, "title": source.title}

        return {
            "search_results": search_results,
            "sources": source_registry,
            "indexed_sources": indexed_sources,
            "processing_stage": "search_complete"
        }

    def generate_search_queries(self, original_query: str) -> List[str]:
        """Generate multiple search queries for comprehensive coverage"""

        prompt = f"""
        You are a search query generator. 
        The user has asked: "{original_query}".

        Your task is to generate exactly 5 diverse and highly relevant search queries 
        that capture different perspectives of the topic.

        Each query should cover a unique angle, following these universal categories:

        1. **Overview** → a broad, general search to understand the topic.  
        2. **Methods/Approaches/Practices** → practical techniques, strategies, or processes.  
        3. **Causes/Factors/Drivers** → key variables, reasons, or influences behind the topic.  
        4. **Impacts/Applications/Examples** → real-world effects, case studies, or domain-specific contexts.  
        5. **Future/Emerging Trends/Challenges** → upcoming developments, open questions, or controversies.

        Guidelines:
        - Queries should be clear, specific, and distinct from each other.  
        - Keep queries concise (7–12 words).  
        - Phrase them naturally, as realistic search queries a researcher would type.  
        - Do not include explanations, only output the queries.
        - Make sure the queries cover different perspectives: causes, effects, methods, comparisons, and case studies where relevant.
        - Ensure at least 2 queries are practical and example-driven.  
        - Ensure at least 2 queries are analytical or comparative.  

        Output format:
        - Return ONLY the list of 5 queries, one per line, in the same order as categories.
        """

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]

        return queries[:5]

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return "unknown"

    def process_search_results(self, results: Any, query: str) -> List[Source]:
        """Convert search results from Tavily into Source objects with substantial content"""
        sources = []

        if isinstance(results, dict) and "results" in results:
            for i, result in enumerate(results["results"]):
                if isinstance(result, dict):
                    content = result.get("content", result.get("snippet", ""))
                    if len(content) < 200 and "snippet" in result:
                        content = result["snippet"]

                    source = Source(
                        id="",  # Empty - let citation manager assign
                        url=result.get("url", ""),
                        title=result.get("title", f"Source {i + 1}"),
                        snippet=content[:800],
                        domain=self.extract_domain(result.get("url", "")),
                        timestamp=datetime.now(),
                        relevance_score=result.get("score", 0.5)
                    )
                    sources.append(source)
        return sources

    def deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on URL"""
        seen_urls = set()
        unique_sources = []

        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        return unique_sources

    def rank_sources(self, sources: List[Source], query: str) -> List[Source]:
        """Rank sources by relevance"""
        # Simple ranking by relevance score and domain authority
        domain_weights = {
            'wikipedia.org': 0.9,
            'reddit.com': 0.7,
            'stackoverflow.com': 0.8,
            'github.com': 0.8,
            'medium.com': 0.7,
        }

        for source in sources:
            domain_weight = domain_weights.get(source.domain, 0.5)
            source.relevance_score = (source.relevance_score + domain_weight) / 2

        return sorted(sources, key=lambda x: x.relevance_score, reverse=True)


