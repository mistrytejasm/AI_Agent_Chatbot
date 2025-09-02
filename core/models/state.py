from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages
from pydantic import BaseModel
from datetime import datetime

class Source(BaseModel):
    id: str
    url: str
    title: str
    snippet: str
    domain: str
    timestamp: datetime
    relevance_score: float

class SearchResult(BaseModel):
    query: str
    sources: List[Source]
    total_results: int
    search_time: datetime

class CitedContent(BaseModel):
    content: str
    source_ids: List[str]
    confidence: float

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    search_results: List[SearchResult]
    sources: Dict[str, Source]
    synthesized_content: List[CitedContent]
    current_query: Optional[str]
    processing_stage: str
