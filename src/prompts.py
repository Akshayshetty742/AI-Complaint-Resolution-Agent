from pathlib import Path
from typing import Dict, Optional, Any
from src.config import PROMPTS_DIR

PROMPT_NAMES = [
    "optimized",
    "structured_rag",
    "few_shot",
    "role_based",
    "zero_shot",
]


class PromptManager:
    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        for name in PROMPT_NAMES:
            file_path = self.prompts_dir / f"{name}.txt"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self._cache[name] = f.read().strip()

    def get_template(self, prompt_type: str = "optimized") -> str:
        """Retrieves raw prompt template by name."""
        if prompt_type not in self._cache:
            file_path = self.prompts_dir / f"{prompt_type}.txt"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self._cache[prompt_type] = f.read().strip()
            else:
                raise ValueError(f"Prompt template '{prompt_type}' not found in {self.prompts_dir}")
        return self._cache[prompt_type]

    def build_prompt(
        self,
        complaint: str,
        context: str = "",
        classification: Optional[Dict[str, Any]] = None,
        prompt_type: str = "optimized"
    ) -> str:
        """Fills complaint, context, and pre-classification metadata into the requested prompt template."""
        template = self.get_template(prompt_type)
        rendered = template.replace("{complaint}", complaint)
        rendered = rendered.replace("{context}", context or "No relevant policy documents found.")

        # Optional pre-classification placeholders
        classification = classification or {}
        rendered = rendered.replace("{category}", str(classification.get("category", "General Support")))
        rendered = rendered.replace("{sentiment}", str(classification.get("sentiment", "Neutral")))
        rendered = rendered.replace("{priority}", str(classification.get("priority", "Medium")))

        return rendered
