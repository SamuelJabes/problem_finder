from pydantic import BaseModel, Field
from typing import Optional, List

class RedditPost(BaseModel):
    """Schema for Reddit post data"""
    title: str
    link: Optional[str] = None
    author: Optional[str] = None
    upvotes: Optional[int] = None
    comment_count: Optional[int] = None
    publish_date: Optional[str] = None
    subreddit: Optional[str] = None