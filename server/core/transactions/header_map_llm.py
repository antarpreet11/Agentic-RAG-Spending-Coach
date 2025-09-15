from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.config import GOOGLE_API_KEY, GEMINI_MODEL
from core.common.cache import cached, set_cached
import orjson
from loguru import logger

class HeaderMap(BaseModel):
    # names must match EXACTLY one of the provided headers (case-sensitive),
    # except *_columns which may list multiple existing headers
    txn_date: str
    posted_date: Optional[str] = None
    amount: str
    description_columns: List[str] = Field(default_factory=list)
    status: Optional[str] = None
    flow_column: Optional[str] = None
    flow_debit_values: List[str] = Field(default_factory=list)
    flow_credit_values: List[str] = Field(default_factory=list)
    currency_literal: Optional[str] = None  # e.g., "CAD" if known

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a meticulous data engineer. Map the CSV headers to a canonical spending schema."),
        ("human", 
         "Headers: {headers}\n\n"
         "Return a JSON that maps to this schema:\n"
         "- txn_date: name of column holding the transaction/original date\n"
         "- posted_date: name of column with posted/settled date, or null\n"
         "- amount: name of the numeric amount column\n"
         "- description_columns: array of 1-3 header names to concatenate for a final description\n"
         "- status: name of the status column (e.g., Posted/Pending), or null\n"
         "- flow_column: name of a column indicating debit/credit, or null if not present\n"
         "- flow_debit_values: which values in flow_column mean debit (e.g., ['Debit'])\n"
         "- flow_credit_values: which values in flow_column mean credit (e.g., ['Credit'])\n"
         "- currency_literal: hardcode 'CAD' if appropriate, else null\n"
         "Only use header names that EXACTLY appear in the list. Do not invent names.")
    ]
)

def map_headers_with_llm(headers: List[str], model_name: Optional[str] = None) -> HeaderMap:
    cache_key = orjson.dumps(sorted(headers)).decode("utf-8")
    
    # Check cache first
    if cached("header_map", cache_key):
        logger.info(f"Using cached header mapping for {len(headers)} headers")
        return HeaderMap.model_validate(cached("header_map", cache_key))
    
    logger.info(f"Calling LLM for header mapping (model: {model_name or GEMINI_MODEL})")
    
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured - cannot use LLM for header mapping")
        raise ValueError("GOOGLE_API_KEY is required for LLM header mapping")

    llm = ChatGoogleGenerativeAI(
        model=model_name or GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY or None,
        temperature=0,
    )

    chain = PROMPT | llm.with_structured_output(HeaderMap)
    
    try:
        result: HeaderMap = chain.invoke({"headers": headers})
        
        logger.info(f"LLM mapping successful: txn_date={result.txn_date}, amount={result.amount}")
        
        # Cache the result
        set_cached("header_map", cache_key, result.model_dump())
        
        return result
        
    except Exception as e:
        logger.error(f"LLM header mapping failed: {str(e)}")
        raise
