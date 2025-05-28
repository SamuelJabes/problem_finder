from pydantic import BaseModel, Field
from typing import Optional

class ClassifySchema(BaseModel):
    """
    Schema for classifying text using OpenAI's model.
    """

    has_problem: str = Field(
        description="""Indicates if the text reveals a user problem point that could be solved. MUST BE either 'YES' or 'NO'.""",
        example=["YES", "NO", "YES"]
    )
    confidence: Optional[int] = Field(None, ge=1, le=10, description="Confidence level from 1-10")
    category: Optional[str] = Field(None, description="Category of the problem")
    intensity: Optional[int] = Field(None, ge=1, le=10, description="Pain intensity from 1-10")