from pydantic import BaseModel, Field
from typing import Optional, List
from schemas.painPoint_schema import PainPoint

class ClusterAnalysisSchema(BaseModel):
    """Schema for cluster analysis by LLM"""
    main_theme: str = Field(..., description="Main theme/problem of this cluster")
    common_problems: List[str] = Field(..., description="List of common problems in this cluster")
    business_opportunity: str = Field(..., description="Potential business opportunity to address these problems")
    target_audience: str = Field(..., description="Target audience for the solution")
    urgency: int = Field(..., ge=1, le=10, description="Urgency level of the problems (1-10)")
    market_size: str = Field(..., description="Estimated market size (small/medium/large)")
    solution_complexity: str = Field(..., description="Solution complexity (simple/medium/complex)")