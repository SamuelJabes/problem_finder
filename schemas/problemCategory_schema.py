from enum import Enum

class ProblemCategory(str, Enum):
    """Predefined categories for pain points"""
    BUSINESS = "business"
    FINANCIAL = "financial" 
    TECHNICAL = "technical"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    HEALTH = "health"
    PRODUCTIVITY = "productivity"
    MARKETING = "marketing"
    CUSTOMER_SERVICE = "customer_service"
    LEGAL = "legal"
    UNKNOWN = "unknown"