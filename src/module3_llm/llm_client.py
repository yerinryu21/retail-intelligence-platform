import os
import sys
import time
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


class RetailLLMClient:
    """
    Centralized LLM client for the Retail Intelligence Platform.
    """

    def __init__(self,
                 model: str = "llama3",
                 temperature: float = 0.1,
                 max_retries: int = 3):

        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries

        print(f"Initializing LLM client: {model}")

        self.llm = OllamaLLM(
            model=model,
            temperature=temperature
        )

        print("LLM client ready")

    def generate(self,
                 prompt: str,
                 max_words: Optional[int] = None) -> str:

        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                cleaned = self._clean_response(response)

                if max_words and len(cleaned.split()) > max_words:
                    words = cleaned.split()
                    truncated = ' '.join(words[:max_words])
                    last_period = max(
                        truncated.rfind('.'),
                        truncated.rfind('!'),
                        truncated.rfind('?')
                    )
                    if last_period > len(truncated) * 0.5:
                        cleaned = truncated[:last_period + 1]
                    else:
                        cleaned = truncated + '...'

                return cleaned

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"LLM attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    print(f"LLM failed after {self.max_retries} attempts: {e}")
                    return self._fallback_response()

    def generate_from_template(self,
                                template: str,
                                variables: dict,
                                max_words: Optional[int] = None) -> str:
        prompt_template = PromptTemplate(
            input_variables=list(variables.keys()),
            template=template
        )
        prompt = prompt_template.format(**variables)
        return self.generate(prompt, max_words=max_words)

    def _clean_response(self, response: str) -> str:
        """
        Clean common LLM response artifacts: markdown, preambles, extra newlines.
        Markdown is stripped first so preamble matching isn't blocked by
        leading ** or ## characters. Preamble matching is case-insensitive
        and ignores leading punctuation/whitespace, so small wording or
        capitalization differences from the model don't slip through.
        """
        cleaned = response.strip()

        # Strip markdown formatting first
        cleaned = cleaned.replace('**', '').replace('*', '')
        cleaned = cleaned.replace('##', '').replace('#', '')
        cleaned = cleaned.strip()

        preambles = [
            "here is", "here's", "here are",
            "based on", "according to",
            "sure,", "certainly,", "of course,",
            "as a retail analytics assistant,",
            "the analysis shows that",
            "i can see that",
            "the summary and recommendation:",
            "summary and recommendation:",
            "summary:",
            "a 3-sentence",
            "here's a 3-sentence",
            "daily inventory briefing:",
        ]

        cleaned_lower = cleaned.lower()
        stripped_something = True
        while stripped_something:
            stripped_something = False
            cleaned_lower = cleaned.lower()
            for preamble in preambles:
                if cleaned_lower.startswith(preamble):
                    rest = cleaned[len(preamble):]
                    rest = rest.lstrip(' ,:-\n')
                    if rest:
                        cleaned = rest[0].upper() + rest[1:]
                    else:
                        cleaned = rest
                    stripped_something = True
                    break
                
        # Generic fallback: if the first line is short and ends with a colon,
        # it's almost certainly a preamble/header the LLM added, regardless
        # of exact wording. Strip it.
        lines = cleaned.split('\n', 1)
        first_line = lines[0].strip()
        if first_line.endswith(':') and len(first_line.split()) <= 8:
            cleaned = lines[1].strip() if len(lines) > 1 else ''

        # Generic fallback 2: a short leading clause ending in a comma
        # (e.g. "The data, ...", "Looking at the data, ...") is also
        # almost always a preamble fragment, not part of the answer.
        comma_idx = cleaned.find(',')
        if 0 < comma_idx <= 40:
            leading_clause = cleaned[:comma_idx].lower()
            if any(w in leading_clause for w in ['data', 'based', 'looking', 'according']):
                rest = cleaned[comma_idx + 1:].lstrip()
                if rest:
                    cleaned = rest[0].upper() + rest[1:]

        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')

        return cleaned.strip()

    def _fallback_response(self) -> str:
        return ("Analysis temporarily unavailable. "
                "Please check that Ollama is running with: ollama serve")

    def test_connection(self) -> bool:
        try:
            response = self.llm.invoke("Reply with exactly one word: working")
            print(f"LLM connection test passed: '{response.strip()}'")
            return True
        except Exception as e:
            print(f"LLM connection test failed: {e}")
            print("Make sure Ollama is running: open a terminal and run 'ollama serve'")
            return False


if __name__ == "__main__":
    client = RetailLLMClient()
    works = client.test_connection()

    if works:
        response = client.generate(
            "In exactly one sentence, explain what customer churn means "
            "for an online retail business.",
            max_words=30
        )
        print(f"\nTest response: {response}")