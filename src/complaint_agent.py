from typing import Dict, Any, Optional
import re

from src.config import EMBEDDING_MODEL, RETRIEVAL_TOP_K
from src.llm import LLMClient
from src.prompts import PromptManager
from src.retriever import PolicyRetriever
from src.complaint_analyzer import ComplaintAnalyzer


class ComplaintResolutionAgent:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[PolicyRetriever] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        self.llm = llm_client or LLMClient()
        self.retriever = retriever or PolicyRetriever(top_k=RETRIEVAL_TOP_K)
        self.prompt_manager = prompt_manager or PromptManager()
        self.analyzer = ComplaintAnalyzer(llm_client=self.llm)

    def _clean_policy_context(self, context_str: str) -> str:
        """
        Removes RAG metadata and document labels from retrieved context
        so that the local fallback response is readable.
        """
        if not context_str:
            return ""

        cleaned = context_str

        # Remove document metadata such as:
        # [Document 1: damaged_product_policy.txt (Relevance Score: 0.479)]
        cleaned = re.sub(
            r"\[Document\s+\d+:\s*.*?\]",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        # Remove relevance-score metadata if present
        cleaned = re.sub(
            r"\(Relevance Score:\s*[\d.]+\)",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        # Remove repeated document/source labels
        cleaned = re.sub(
            r"\[RAG SOURCE:.*?\]",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _build_fallback_response(
        self,
        complaint: str,
        analysis: Dict[str, Any],
        retrieved_chunks: list,
        context_str: str,
        reason: str = "LLM unavailable"
    ) -> Dict[str, Any]:

        category = analysis["category"]
        sentiment = analysis["sentiment"]
        priority = analysis["priority"]

        if retrieved_chunks:
            sources = []

            for chunk in retrieved_chunks:
                source = chunk.get("source", "company policy")

                if source not in sources:
                    sources.append(source)

            source_name = sources[0] if sources else "company policy"

            # Clean the retrieved RAG context
            policy_text = self._clean_policy_context(context_str)

            # Keep fallback response concise
            if len(policy_text) > 650:
                policy_text = policy_text[:650].rsplit(" ", 1)[0] + "..."

            resolution = (
                f"Based on the retrieved policy from {source_name}, "
                f"the complaint should be handled according to the applicable "
                f"{category.lower()} procedure. "
                f"{policy_text}"
            )

            customer_response = (
                f"Thank you for contacting customer support regarding your "
                f"{category.lower()} complaint. "
                f"We have reviewed the applicable company policy and identified "
                f"the relevant next steps. Your request will be processed "
                f"according to the policy, and our support team will contact "
                f"you if any additional verification is required."
            )

        else:
            resolution = (
                f"No relevant policy document was retrieved for this "
                f"{category.lower()} complaint. Manual review is required."
            )

            customer_response = (
                "Thank you for contacting customer support. "
                "We were unable to retrieve the relevant policy information, "
                "so your request has been forwarded for manual review."
            )

        return {
            "category": category,
            "sentiment": sentiment,
            "priority": priority,
            "resolution": resolution,
            "customer_response": customer_response,
            "retrieved_chunks": retrieved_chunks,
            "technical_details": {
                "embedding_model": EMBEDDING_MODEL,
                "retrieved_chunks_count": len(retrieved_chunks),
                "retrieval_top_k": self.retriever.top_k,
                "prompt_type": "fallback",
                "llm_model": self.llm.model,
                "execution_mode": f"Local Fallback ({reason})"
            },
            "api_configured": False,
            "is_valid_json": True,
            "prompt_used": "fallback"
        }

    def resolve(
        self,
        complaint: str,
        prompt_type: str = "optimized"
    ) -> Dict[str, Any]:

        complaint = complaint.strip()

        if not complaint:
            raise ValueError("Complaint cannot be empty.")

        # ---------------------------------------------------------
        # STEP 1: Complaint Analysis
        # ---------------------------------------------------------
        initial_analysis = self.analyzer.analyze(complaint)

        # ---------------------------------------------------------
        # STEP 2: RAG Retrieval
        # ---------------------------------------------------------
        context_str, retrieved_chunks = (
            self.retriever.get_context_for_complaint(complaint)
        )

        # ---------------------------------------------------------
        # STEP 3: Check LLM Configuration
        # ---------------------------------------------------------
        is_live_llm = self.llm.is_configured()

        mode_label = (
            "Live LLM (OpenAI-compatible)"
            if is_live_llm
            else "Local Fallback (No API Key)"
        )

        tech_details = {
            "embedding_model": EMBEDDING_MODEL,
            "retrieved_chunks_count": len(retrieved_chunks),
            "retrieval_top_k": self.retriever.top_k,
            "prompt_type": prompt_type,
            "llm_model": self.llm.model,
            "execution_mode": mode_label
        }

        # ---------------------------------------------------------
        # STEP 4: Local Fallback if No API Key
        # ---------------------------------------------------------
        if not is_live_llm:
            return self._build_fallback_response(
                complaint=complaint,
                analysis=initial_analysis,
                retrieved_chunks=retrieved_chunks,
                context_str=context_str,
                reason="No API Key"
            )

        # ---------------------------------------------------------
        # STEP 5: Build RAG Prompt
        # ---------------------------------------------------------
        prompt = self.prompt_manager.build_prompt(
            complaint=complaint,
            context=context_str,
            classification=initial_analysis,
            prompt_type=prompt_type
        )

        # ---------------------------------------------------------
        # STEP 6: Call Live LLM
        # ---------------------------------------------------------
        try:
            raw_output = self.llm.generate(prompt)

            structured_data = self.llm.extract_json(raw_output)

            is_valid_json = self.llm.validate_structured_output(
                structured_data
            )

            # -----------------------------------------------------
            # STEP 7: Handle Invalid LLM JSON
            # -----------------------------------------------------
            if not is_valid_json:

                category = (
                    structured_data.get("category")
                    if structured_data
                    else None
                ) or initial_analysis["category"]

                sentiment = (
                    structured_data.get("sentiment")
                    if structured_data
                    else None
                ) or initial_analysis["sentiment"]

                priority = (
                    structured_data.get("priority")
                    if structured_data
                    else None
                ) or initial_analysis["priority"]

                resolution = (
                    structured_data.get("resolution")
                    if structured_data
                    else None
                ) or "Manual review required."

                customer_resp = (
                    structured_data.get("customer_response")
                    if structured_data
                    else None
                ) or raw_output

            else:
                category = structured_data["category"]
                sentiment = structured_data["sentiment"]
                priority = structured_data["priority"]
                resolution = structured_data["resolution"]
                customer_resp = structured_data["customer_response"]

            return {
                "category": category,
                "sentiment": sentiment,
                "priority": priority,
                "resolution": resolution,
                "customer_response": customer_resp,
                "retrieved_chunks": retrieved_chunks,
                "technical_details": tech_details,
                "api_configured": True,
                "is_valid_json": is_valid_json,
                "prompt_used": prompt_type
            }

        # ---------------------------------------------------------
        # STEP 8: Automatic Fallback if Gemini/API Fails
        # ---------------------------------------------------------
        except Exception:
            return self._build_fallback_response(
                complaint=complaint,
                analysis=initial_analysis,
                retrieved_chunks=retrieved_chunks,
                context_str=context_str,
                reason="LLM/API unavailable"
            )