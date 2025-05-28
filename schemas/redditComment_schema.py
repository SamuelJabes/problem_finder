from pydantic import BaseModel, Field
from typing import Optional, List

class RedditComment(BaseModel):
    """Schema for Reddit comment data"""
    body: str
    author: Optional[str] = None
    upvotes: Optional[int] = None
    publish_date: Optional[str] = None
    post_link: Optional[str] = None