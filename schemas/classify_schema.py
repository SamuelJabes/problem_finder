from pydantic import BaseModel, Field
from typing import List

class ClassifySchema(BaseModel):
    """
    Schema for classifying text using OpenAI's model.
    """

    has_problem: List[str] = Field(
        ...,
        description="Indicates if the text reveals a user problem point that could be solved. Must be either 'yes' or 'no'."
    )