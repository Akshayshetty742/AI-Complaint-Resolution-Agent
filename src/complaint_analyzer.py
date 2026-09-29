import json
from typing import Dict, Any
from src.llm import LLMClient
from src.config import CATEGORIES, SENTIMENTS, PRIORITIES

ANALYSIS_PROMPT = """Analyze the following customer complaint and classify it into:
- Category (Must be exactly one of: Refund, Return, Delivery, Damaged Product, Warranty, Payment, General Support)
- Sentiment (Must be exactly one of: Positive, Neutral, Negative)
- Priority (Must be exactly one of: Low, Medium, High)

Customer Complaint:
"{complaint}"

Return ONLY a valid JSON object without markdown fences:
{{
  "category": "Refund | Return | Delivery | Damaged Product | Warranty | Payment | General Support",
  "sentiment": "Positive | Neutral | Negative",
  "priority": "Low | Medium | High"
}}"""


class ComplaintAnalyzer:
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def analyze(self, complaint: str) -> Dict[str, str]:
        """Performs initial complaint classification (category, sentiment, priority) using LLM."""
        if self.llm.is_configured():
            try:
                prompt = ANALYSIS_PROMPT.format(complaint=complaint)
                raw_response = self.llm.generate(prompt, temperature=0.0)
                parsed = self.llm.extract_json(raw_response)
                if parsed and "category" in parsed:
                    # Sanitize values
                    category = self._match_choice(parsed.get("category", ""), CATEGORIES, "General Support")
                    sentiment = self._match_choice(parsed.get("sentiment", ""), SENTIMENTS, "Negative")
                    priority = self._match_choice(parsed.get("priority", ""), PRIORITIES, "Medium")
                    return {
                        "category": category,
                        "sentiment": sentiment,
                        "priority": priority
                    }
            except Exception as e:
                print(f"LLM analysis warning: {e}. Falling back to heuristic classifier.")

        # Heuristic fallback if LLM is not configured or fails
        return self._heuristic_analysis(complaint)

    @staticmethod
    def _match_choice(val: str, valid_choices: list, default: str) -> str:
        for choice in valid_choices:
            if choice.lower() in val.lower():
                return choice
        return default

    @staticmethod
    def _heuristic_analysis(complaint: str) -> Dict[str, str]:
        """Lightweight heuristic fallback when LLM is unavailable."""
        text = complaint.lower()

        # Category scoring rules: (category, weight, keywords/phrases)
        scores = {
            "Payment": 0,
            "Refund": 0,
            "Return": 0,
            "Damaged Product": 0,
            "Warranty": 0,
            "Delivery": 0,
            "General Support": 0,
        }

        # 1. Payment indicators
        payment_phrases = [
            "payment was deducted", "money deducted", "amount deducted",
            "not confirmed", "order was not confirmed", "order not confirmed",
            "double charge", "charged twice", "duplicate charge", "unauthorized charge",
            "payment failed", "payment failure", "transaction failed"
        ]
        payment_words = ["payment", "deducted", "debited", "billing", "credit card", "debit card", "upi", "net banking", "transaction"]
        for p in payment_phrases:
            if p in text:
                scores["Payment"] += 3
        for w in payment_words:
            if w in text:
                scores["Payment"] += 1

        # 2. Refund indicators
        refund_phrases = [
            "haven't received my refund", "cancelled my order", "order cancelled",
            "cancel my order", "money back", "cancelled prior to dispatch",
            "money be returned", "money returned to my card", "money returned"
        ]
        refund_words = ["refund", "refunds", "refunded", "reimbursement"]
        for p in refund_phrases:
            if p in text:
                scores["Refund"] += 3
        for w in refund_words:
            if w in text:
                scores["Refund"] += 2

        # 3. Damaged Product indicators
        damage_phrases = [
            "arrived damaged", "received damaged", "broken out of the box",
            "defective out-of-the-box", "damaged item", "out of the box",
            "shattered inside", "crack right out of the box"
        ]
        damage_words = ["damaged", "broken", "cracked", "crack", "dent", "doa", "smashed", "shattered"]
        for p in damage_phrases:
            if p in text:
                scores["Damaged Product"] += 3
        for w in damage_words:
            if w in text:
                scores["Damaged Product"] += 2

        # 4. Warranty indicators
        warranty_phrases = [
            "covered under warranty", "stopped working", "stopped functioning",
            "after two months", "after a month", "after four months", "months of purchase",
            "normal use", "motor failed"
        ]
        warranty_words = ["warranty", "guarantee", "repair", "service center", "hardware malfunction", "malfunction", "months"]
        for p in warranty_phrases:
            if p in text:
                scores["Warranty"] += 3
        for w in warranty_words:
            if w in text:
                scores["Warranty"] += 2

        # 5. Delivery indicators
        delivery_phrases = ["when will my order be delivered", "supposed to arrive", "lost in transit", "track order", "estimated delivery"]
        delivery_words = ["delivery", "delivered", "shipping", "courier", "arrive", "tracking", "transit", "dispatch"]
        for p in delivery_phrases:
            if p in text:
                scores["Delivery"] += 3
        for w in delivery_words:
            if w in text:
                scores["Delivery"] += 1

        # 6. Return indicators
        return_phrases = ["return period", "return window", "send back", "doorstep pickup"]
        return_words = ["return", "returns", "returning", "exchange"]
        for p in return_phrases:
            if p in text:
                scores["Return"] += 3
        for w in return_words:
            if w in text:
                scores["Return"] += 2

        # 7. General Support indicators
        support_phrases = ["help changing", "customer care", "customer service", "general support"]
        support_words = ["address", "account", "profile", "password", "login", "email address", "phone number"]
        for p in support_phrases:
            if p in text:
                scores["General Support"] += 3
        for w in support_words:
            if w in text:
                scores["General Support"] += 2

        # Determine highest scoring category
        best_category, max_score = max(scores.items(), key=lambda item: item[1])
        category = best_category if max_score > 0 else "General Support"

        # Sentiment detection
        if any(w in text for w in ["angry", "terrible", "worst", "broken", "fraud", "stole", "never", "hate", "haven't", "still", "not confirmed", "failed", "delay"]):
            sentiment = "Negative"
        elif any(w in text for w in ["thank", "great", "please", "appreciate"]):
            sentiment = "Neutral"
        else:
            sentiment = "Neutral"

        # Priority detection
        if any(w in text for w in ["damaged", "broken", "fraud", "urgent", "emergency", "immediately", "stolen", "twice", "not confirmed"]):
            priority = "High"
        elif any(w in text for w in ["refund", "warranty", "delay", "lost", "five days", "two months"]):
            priority = "Medium"
        else:
            priority = "Low"

        return {
            "category": category,
            "sentiment": sentiment,
            "priority": priority
        }
