"""
Transaction schema definitions
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class Txn(BaseModel):
    """Transaction model for validation"""
    
    txn_id: str = Field(description="Unique transaction ID")
    account_id: str = Field(description="Account identifier")
    source_bank: str = Field(description="Source bank name")
    txn_date: Optional[str] = Field(description="Transaction date")
    posted_date: Optional[str] = Field(description="Posted date")
    description: str = Field(description="Transaction description")
    status: Optional[str] = Field(description="Transaction status")
    flow: Optional[str] = Field(description="Transaction flow (debit/credit)")
    amount: float = Field(description="Transaction amount")
    currency: str = Field(default="CAD", description="Currency code")
    
    class Config:
        extra = "allow"  # Allow additional fields
