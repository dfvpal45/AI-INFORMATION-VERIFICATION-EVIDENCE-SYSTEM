from pydantic import BaseModel, Field
from typing import Optional

class VerifyTextRequest(BaseModel):
    text: str = Field(min_length=3, max_length=30000)
    source_url: Optional[str] = None

class FeedbackRequest(BaseModel):
    feedback: str = Field(min_length=1, max_length=3000)
