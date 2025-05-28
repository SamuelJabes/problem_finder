from pydantic import BaseModel, Field
from typing import Optional, List

class PainPoint(BaseModel):
    """Schema for identified pain point"""
    text: str
    has_problem: str
    confidence: Optional[int] = None
    category: Optional[str] = None
    intensity: Optional[int] = None
    source: Optional[str] = None  # post or comment
    subreddit: Optional[str] = None
    author: Optional[str] = None
    upvotes: Optional[int] = None