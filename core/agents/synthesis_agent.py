from typing import List, Dict, Any
from langchain_groq import ChatGroq
from core.models.state import AgentState, CitedContent
from core.agents.citation_manager import CitationManager


class SynthesisAgent:
    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b")
        self.citation_manager = CitationManager()
        self.citation_manager.reset()

    def synthesize_response(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize information from sources into a comprehensive response"""
        self.citation_manager.reset()
        query = state["current_query"]
        sources = state["sources"]

        # Load sources fresh (dedup happens here via add_source)
        self.citation_manager.source_registry = {}
        for source in sources.values():  # Re-add to ensure dedup
            self.citation_manager.add_source(source)

        # create context from source
        context_text = self.build_context_from_sources(self.citation_manager.source_registry)


        # Generate synthesized response
        synthesis_prompt = f"""You are an expert AI assistant providing comprehensive, authoritative answers like Perplexity AI.

        Guidelines for response quality:
        1. **Structure**: Start with a clear overview, then dive into detailed explanations with subheadings
        2. **Depth**: Provide thorough explanations with context, not just lists
        3. **Authority**: Write confidently using technical details and examples
        4. **Flow**: Create smooth transitions between concepts
        5. **Citations**: Use [1], [2] etc. inline after specific facts (SQUARE brackets only)
        6. **Length**: Aim for comprehensive coverage - 800-1500 words is ideal
        7. **Examples**: Include concrete examples and use cases where relevant

        Response structure to follow:
        - Opening paragraph explaining the concept
        - ## What is Overfitting? (detailed explanation with examples)
        - ## Why Overfitting Occurs (root causes with context)
        - ## Techniques to Prevent Overfitting (detailed explanations of each method)
        - ## Best Practices and Recommendations (synthesis and practical advice)

        Write like a domain expert explaining to someone who wants to truly understand the topic.

        User query: {query}
        Sources context: {context_text}
        """

        response = self.llm.invoke([{"role": "user", "content": synthesis_prompt}])
        synthesized_text = response.content

        # Create cited content (limit to top sources)
        top_source_ids = list(self.citation_manager.source_registry.keys())[:5]  # Cap for citation
        cited_content = self.citation_manager.create_cited_content(synthesized_text, top_source_ids)

        final_response = self.format_final_response(cited_content, self.citation_manager)
        return {
            "synthesized_content": [cited_content],
            "final_response": final_response,
            "processing_stage": "synthesis_complete"
        }

    def build_context_from_sources(self, sources: Dict[str, Any]) -> str:
        """Build detailed context string from sources"""
        context = "SOURCES FOR SYNTHESIS:\n\n"
        for i, (source_id, source) in enumerate(sources.items(), 1):
            context += f"## Source {i} - {source.domain}\n"
            context += f"**Title:** {source.title}\n"
            context += f"**Content:** {source.snippet}\n"
            context += f"**URL:** {source.url}\n"
            context += f"**Relevance Score:** {source.relevance_score:.2f}\n"
            context += "-" * 80 + "\n\n"

        context += "SYNTHESIS INSTRUCTIONS: Use this information to create a comprehensive, detailed response that covers all aspects of the query. Draw connections between sources and provide deep explanations.\n"
        return context

    def format_final_response(self, cited_content: CitedContent, citation_manager: CitationManager) -> str:
        """Format the final response with sources"""
        response = cited_content.content
        response += self.citation_manager.format_sources_list()
        return response

