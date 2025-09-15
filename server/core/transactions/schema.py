from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date
import re

class Txn(BaseModel):
    txn_id: str
    account_id: str
    source_bank: str
    txn_date: date
    posted_date: Optional[date] = None
    description: str
    status: Optional[str] = None
    flow: Literal["debit","credit"]
    amount: float
    currency: str = "CAD"

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float):
        if v <= 0:
            raise ValueError("amount must be positive magnitude after normalization")
        return v

    @field_validator("description")
    @classmethod
    def non_empty_desc(cls, v: str):
        if not (v and v.strip()):
            raise ValueError("description required")
        return v.strip()
