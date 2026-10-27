from sqlalchemy import String, Text, Float, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class VerificationRecord(Base):
    __tablename__ = "verification_records"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    input_type: Mapped[str] = mapped_column(String(30), default="text")
    original_input: Mapped[str] = mapped_column(Text)
    claims: Mapped[dict] = mapped_column(JSON, default=list)
    verdict: Mapped[str] = mapped_column(String(40), default="Unverified")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    user_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
