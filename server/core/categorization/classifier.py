"""
Transaction Categorization System
Multi-layer approach: Rules + Embeddings + LLM
"""

from typing import Dict, List, Tuple, Optional
import yaml
import re
from pathlib import Path
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class CategorizationResult(BaseModel):
    """Pydantic model for LLM categorization response"""
    category: str = Field(description="Category key from taxonomy")
    subcategory: str = Field(description="Subcategory name")
    confidence: float = Field(description="Confidence score between 0 and 1")

class BatchCategorizationResult(BaseModel):
    """Pydantic model for batch LLM categorization response"""
    transactions: List[CategorizationResult] = Field(description="List of categorization results")

class TaxonomyManager:
    """Manages the categorization taxonomy"""
    
    def __init__(self, taxonomy_path: str):
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy()
    
    def _load_taxonomy(self) -> Dict:
        """Load taxonomy from YAML file"""
        try:
            with open(self.taxonomy_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            return {}
    
    def get_categories(self) -> List[str]:
        """Get list of all category keys"""
        return list(self.taxonomy.get('categories', {}).keys())
    
    def get_category_name(self, category_key: str) -> str:
        """Get display name for category"""
        return self.taxonomy.get('categories', {}).get(category_key, {}).get('name', category_key)
    
    def get_subcategories(self, category_key: str) -> List[str]:
        """Get subcategories for a category"""
        return self.taxonomy.get('categories', {}).get(category_key, {}).get('subcategories', [])
    
    def get_merchant_dictionary(self) -> Dict[str, List[str]]:
        """Get merchant dictionary"""
        return self.taxonomy.get('merchant_dictionary', {})
    
    def get_regex_patterns(self) -> Dict[str, List[str]]:
        """Get regex patterns"""
        return self.taxonomy.get('regex_patterns', {})
    
    def get_mcc_codes(self) -> Dict[str, List[str]]:
        """Get MCC code mappings"""
        return self.taxonomy.get('mcc_codes', {})

class RulesClassifier:
    """Rule-based transaction classifier"""
    
    def __init__(self, taxonomy_manager: TaxonomyManager):
        self.taxonomy = taxonomy_manager
        self.merchant_dict = self.taxonomy.get_merchant_dictionary()
        self.regex_patterns = self.taxonomy.get_regex_patterns()
        self.mcc_codes = self.taxonomy.get_mcc_codes()
    
    def classify(self, description: str, amount: float, mcc_code: Optional[str] = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Classify transaction using rules
        Returns: (category, subcategory, confidence)
        """
        description_upper = description.upper()
        
        # 1. Check merchant dictionary (exact match)
        for merchant, (category, subcategory) in self.merchant_dict.items():
            if merchant in description_upper:
                return category, subcategory, 0.95
        
        # 2. Check regex patterns
        for pattern, (category, subcategory) in self.regex_patterns.items():
            if re.search(pattern, description_upper):
                return category, subcategory, 0.90
        
        # 3. Check MCC codes
        if mcc_code and mcc_code in self.mcc_codes:
            category, subcategory = self.mcc_codes[mcc_code]
            return category, subcategory, 0.85
        
        # 4. Amount-based rules
        if amount < 0.50:
            return "transfers", "ATM Withdrawal", 0.70
        
        return None, None, 0.0
    
    def add_merchant(self, merchant: str, category: str, subcategory: str):
        """Add new merchant to dictionary"""
        self.merchant_dict[merchant] = [category, subcategory]
        logger.info(f"Added merchant: {merchant} -> {category}/{subcategory}")

class EmbeddingsClassifier:
    """Embeddings-based KNN classifier"""
    
    def __init__(self, taxonomy_manager: TaxonomyManager):
        self.taxonomy = taxonomy_manager
        self.embeddings_db = None  # Will be initialized with FAISS/ChromaDB
        self.seed_data = []  # Labeled transaction data
    
    def initialize(self, seed_data_path: Optional[str] = None):
        """Initialize embeddings database with seed data"""
        if seed_data_path:
            self._load_seed_data(seed_data_path)
        
        # TODO: Implement FAISS/ChromaDB initialization
        logger.info("Embeddings classifier initialized")
    
    def _load_seed_data(self, seed_data_path: str):
        """Load seed data for training"""
        # TODO: Load labeled transactions
        pass
    
    def classify(self, description: str, amount: float) -> Tuple[Optional[str], Optional[str], float]:
        """
        Classify using embeddings KNN
        Returns: (category, subcategory, confidence)
        """
        # TODO: Implement embeddings-based classification
        return None, None, 0.0
    
    def add_training_data(self, description: str, category: str, subcategory: str):
        """Add new training data"""
        self.seed_data.append({
            'description': description,
            'category': category,
            'subcategory': subcategory
        })

class LLMClassifier:
    """LLM-based transaction classifier using Google Gemini"""
    
    def __init__(self, taxonomy_manager: TaxonomyManager):
        self.taxonomy = taxonomy_manager
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Gemini LLM client"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from core.config import GOOGLE_API_KEY, GEMINI_MODEL
            
            if not GOOGLE_API_KEY:
                logger.error("GEMINI_API_KEY not configured for LLM categorization")
                self.llm = None
                return
            
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,  # Low temperature for consistent categorization
            )
            
            # Create the categorization prompt
            self.prompt = self._create_categorization_prompt()
            logger.info("LLM classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM classifier: {e}")
            self.llm = None
    
    def _create_categorization_prompt(self) -> ChatPromptTemplate:
        """Create the categorization prompt"""
        
        # Get all categories and subcategories
        categories_text = ""
        for category_key, category_data in self.taxonomy.taxonomy.get('categories', {}).items():
            category_name = category_data.get('name', category_key)
            subcategories = category_data.get('subcategories', [])
            subcategories_text = ", ".join(subcategories)
            categories_text += f"- {category_key} ({category_name}): {subcategories_text}\n"
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor categorizing transactions for spending analysis.

Your task is to categorize transactions into the most appropriate category and subcategory based on the transaction description, amount, and context.

Available Categories:
{categories}

IMPORTANT: Use the EXACT category keys (lowercase with underscores) from the list above, NOT the display names.

Guidelines:
1. Choose the MOST SPECIFIC subcategory that fits the transaction
2. Consider words in the description to align with likeliness of category
3. Consider the amount - small amounts might be fees, large amounts might be major purchases
4. Be consistent with similar transactions

Return your response as JSON with this exact format:
{{
    "category": "category_key",
    "subcategory": "Subcategory Name",
    "confidence": 0.85
}}"""),
            ("human", """Transaction Details:
Description: {description}
Amount: ${amount}
Card: {card_name}
Date: {date}

Categorize this transaction:""")
        ])
        
        return prompt_template.partial(categories=categories_text)
    
    def classify(self, description: str, amount: float, card_name: str, date: str) -> Tuple[str, str, float, str]:
        """
        Classify using LLM
        Returns: (category, subcategory, confidence, reasoning)
        """
        if not self.llm:
            return "other", "Miscellaneous", 0.3, "LLM not available"
        
        try:
            # Create the chain
            chain = self.prompt | self.llm.with_structured_output(CategorizationResult)
            
            # Invoke the LLM
            result: CategorizationResult = chain.invoke({
                "description": description,
                "amount": amount,
                "card_name": card_name,
                "date": date
            })
            
            # Extract results
            category = result.category
            subcategory = result.subcategory
            confidence = result.confidence
            
            # Validate category exists in taxonomy
            if category not in self.taxonomy.get_categories():
                logger.warning(f"LLM returned invalid category: {category}, defaulting to 'other'")
                category = "other"
                subcategory = "Miscellaneous"
                confidence = 0.5
            
            logger.info(f"LLM categorized: {description[:30]}... -> {category}/{subcategory} (confidence: {confidence:.2f})")
            
            return category, subcategory, confidence, "LLM classification"
            
        except Exception as e:
            logger.error(f"LLM categorization failed: {e}")
            return "other", "Miscellaneous", 0.3, f"LLM error: {str(e)}"

class BatchedLLMClassifier:
    """Batched LLM classifier for processing multiple transactions efficiently"""
    
    def __init__(self, taxonomy_manager: TaxonomyManager):
        self.taxonomy = taxonomy_manager
        self.llm = None
        self.prompt = None
        self._initialize_llm()
        if self.llm:
            self.prompt = self._create_batch_prompt()
    
    def _initialize_llm(self):
        """Initialize the LLM client"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from core.config import GOOGLE_API_KEY, GEMINI_MODEL
            
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY or None,
                temperature=0,
            )
            logger.info("Batched LLM classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize batched LLM: {e}")
            self.llm = None
    
    def _create_batch_prompt(self) -> ChatPromptTemplate:
        """Create the batch categorization prompt"""
        
        # Get all categories and subcategories
        categories_text = ""
        for category_key, category_data in self.taxonomy.taxonomy.get('categories', {}).items():
            category_name = category_data.get('name', category_key)
            subcategories = category_data.get('subcategories', [])
            subcategories_text = ", ".join(subcategories)
            categories_text += f"- {category_key} ({category_name}): {subcategories_text}\n"
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor categorizing multiple transactions for spending analysis.

Your task is to categorize each transaction into the most appropriate category and subcategory based on the transaction description, amount, and context.

Available Categories:
{categories}

IMPORTANT: Use the EXACT category keys (lowercase with underscores) from the list above, NOT the display names.

Guidelines:
1. Choose the MOST SPECIFIC subcategory that fits each transaction
2. Ensure that keywords in the description are taken into account for choosing category
3. Consider the amount - small amounts might be fees, large amounts might be major purchases
4. Be consistent with similar transactions
5. ALWAYS use the exact category keys from the list above - never invent new ones

Return your response as JSON with this exact format:
{{
    "transactions": [
        {{
            "category": "category_key",
            "subcategory": "Subcategory Name",
            "confidence": 0.85
        }},
        {{
            "category": "category_key",
            "subcategory": "Subcategory Name",
            "confidence": 0.90
        }}
    ]
}}

Process ALL {batch_size} transactions in the batch. Return exactly {batch_size} results."""),
            ("human", """Batch of Transactions to Categorize:

{transactions}

Categorize each transaction in the batch:""")
        ])
        
        return prompt_template.partial(categories=categories_text)
    
    def classify_batch(self, transactions: List[Dict]) -> List[Tuple[str, str, float, str]]:
        """
        Classify a batch of transactions using LLM
        Returns: List of (category, subcategory, confidence, reasoning) tuples
        """
        if not self.llm or not transactions:
            return [("other", "Miscellaneous", 0.3, "LLM not available")] * len(transactions)
        
        try:
            # Format transactions for the prompt
            transactions_text = ""
            for i, txn in enumerate(transactions, 1):
                transactions_text += f"{i}. Description: {txn.get('description', '')}\n"
                transactions_text += f"   Amount: ${txn.get('amount', 0.0)}\n"
                transactions_text += f"   Card: {txn.get('card_name', '')}\n"
                transactions_text += f"   Date: {txn.get('date', '')}\n\n"
            
            # Create the chain
            chain = self.prompt | self.llm.with_structured_output(BatchCategorizationResult)
            
            # Invoke the LLM
            result: BatchCategorizationResult = chain.invoke({
                "transactions": transactions_text,
                "batch_size": len(transactions)
            })
            
            # Extract and validate results
            results = []
            for i, categorization in enumerate(result.transactions):
                category = categorization.category
                subcategory = categorization.subcategory
                confidence = categorization.confidence
                
                # Validate category exists in taxonomy
                if category not in self.taxonomy.get_categories():
                    logger.warning(f"LLM returned invalid category: {category}, defaulting to 'other'")
                    category = "other"
                    subcategory = "Miscellaneous"
                    confidence = 0.5
                
                results.append((category, subcategory, confidence, "Batched LLM classification"))
            
            logger.info(f"Batched LLM categorized {len(transactions)} transactions in 1 API call")
            return results
            
        except Exception as e:
            logger.error(f"Batched LLM categorization failed: {e}")
            return [("other", "Miscellaneous", 0.3, f"LLM error: {str(e)}")] * len(transactions)

class EnsembleClassifier:
    """Ensemble classifier combining all approaches"""
    
    def __init__(self, taxonomy_path: str):
        self.taxonomy = TaxonomyManager(taxonomy_path)
        self.rules_classifier = RulesClassifier(self.taxonomy)
        self.embeddings_classifier = EmbeddingsClassifier(self.taxonomy)
        self.llm_classifier = LLMClassifier(self.taxonomy)
        self.batched_llm_classifier = BatchedLLMClassifier(self.taxonomy)
        
        # Configuration
        self.confidence_threshold = 0.8
        self.llm_fallback_threshold = 0.6
        self.batch_size = 50  # Process 50 transactions per LLM call
    
    def classify_transaction(self, description: str, amount: float, card_name: str, date: str, mcc_code: Optional[str] = None) -> Dict:
        """
        Classify a single transaction using ensemble approach
        Returns: {
            'category': str,
            'subcategory': str,
            'confidence': float,
            'method': str,
            'reasoning': str
        }
        """
        # 1. Try rules-based classification
        category, subcategory, confidence = self.rules_classifier.classify(description, amount, mcc_code)
        if confidence >= self.confidence_threshold:
            return {
                'category': category,
                'subcategory': subcategory,
                'confidence': confidence,
                'method': 'rules',
                'reasoning': f"Matched merchant/regex pattern"
            }
        
        # 2. Try embeddings-based classification
        category, subcategory, confidence = self.embeddings_classifier.classify(description, amount)
        if confidence >= self.confidence_threshold:
            return {
                'category': category,
                'subcategory': subcategory,
                'confidence': confidence,
                'method': 'embeddings',
                'reasoning': f"KNN similarity match"
            }
        
        # 3. Fallback to LLM
        if confidence < self.llm_fallback_threshold:
            category, subcategory, confidence, reasoning = self.llm_classifier.classify(
                description, amount, card_name, date
            )
            return {
                'category': category,
                'subcategory': subcategory,
                'confidence': confidence,
                'method': 'llm',
                'reasoning': reasoning
            }
        
        # 4. Default fallback
        return {
            'category': 'other',
            'subcategory': 'Miscellaneous',
            'confidence': 0.3,
            'method': 'fallback',
            'reasoning': 'No confident classification found'
        }
    
    def classify_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Classify multiple transactions (legacy method - use classify_batch_efficient instead)"""
        results = []
        for txn in transactions:
            result = self.classify_transaction(
                description=txn.get('description', ''),
                amount=txn.get('amount', 0.0),
                card_name=txn.get('account', ''),
                date=txn.get('txn_date', ''),
                mcc_code=txn.get('mcc_code')
            )
            results.append({**txn, **result})
        return results
    
    def classify_batch_efficient(self, transactions: List[Dict]) -> List[Dict]:
        """
        Efficiently classify multiple transactions using batched LLM calls
        This method reduces API calls by batching transactions that need LLM processing
        """
        results = []
        llm_batch = []
        
        # First pass: Try rules and embeddings for each transaction
        for txn in transactions:
            description = txn.get('description', '')
            amount = txn.get('amount', 0.0)
            card_name = txn.get('card_name', txn.get('account', ''))
            date = txn.get('date', txn.get('txn_date', ''))
            mcc_code = txn.get('mcc_code')
            
            # Try rules-based classification
            category, subcategory, confidence = self.rules_classifier.classify(description, amount, mcc_code)
            if confidence >= self.confidence_threshold:
                results.append({
                    **txn,
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': confidence,
                    'method': 'rules',
                    'reasoning': f"Matched merchant/regex pattern"
                })
                continue
            
            # Try embeddings-based classification
            category, subcategory, confidence = self.embeddings_classifier.classify(description, amount)
            if confidence >= self.confidence_threshold:
                results.append({
                    **txn,
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': confidence,
                    'method': 'embeddings',
                    'reasoning': f"KNN similarity match"
                })
                continue
            
            # Add to LLM batch for later processing
            llm_batch.append({
                'description': description,
                'amount': amount,
                'card_name': card_name,
                'date': date,
                'original_txn': txn
            })
        
        # Process LLM batches
        if llm_batch:
            logger.info(f"Processing {len(llm_batch)} transactions with batched LLM calls")
            
            # Split into batches of batch_size
            for i in range(0, len(llm_batch), self.batch_size):
                batch = llm_batch[i:i + self.batch_size]
                
                # Process batch with LLM
                batch_results = self.batched_llm_classifier.classify_batch(batch)
                
                # Add results to final results
                for j, (category, subcategory, confidence, reasoning) in enumerate(batch_results):
                    original_txn = batch[j]['original_txn']
                    results.append({
                        **original_txn,
                        'category': category,
                        'subcategory': subcategory,
                        'confidence': confidence,
                        'method': 'llm_batch',
                        'reasoning': reasoning
                    })
        
        return results
