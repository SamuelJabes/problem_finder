from pydantic import BaseModel, Field
from typing import List

class ClassifySchema(BaseModel):
    """
    Schema for classifying text using OpenAI's model.
    """

    has_problem: str = Field(
        description="""Indicates if the text reveals a user problem point that could be solved. MUST BE either 'YES' or 'NO'.""",
        example=["YES", "NO", "YES"]
    )