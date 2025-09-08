import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.llm_manager import AzureAIClientManager

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=2000, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant for insurance policy document summarization optimized for search retrieval. "
                        "Focus on preserving key details, conditions, numbers, exceptions, and include diverse keywords for better searchability.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a comprehensive summary of the following insurance policy text, "
                        f"preserving all important details and including relevant keywords for search optimization: {context}",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=2000, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant for insurance policy document summarization optimized for search retrieval. "
                        "Focus on preserving key details, conditions, numbers, exceptions, and include diverse keywords for better searchability.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a comprehensive summary of the following insurance policy text, "
                        f"preserving all important details and including relevant keywords for search optimization: {context}",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class AzureSummarizationModel(BaseSummarizationModel):
    """
    Summarization using Azure OpenAI (e.g., gpt-4o) via AzureAIClientManager.
    """

    def __init__(self, client: AzureAIClientManager):
        self.client = client

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context: str, max_tokens: int = 4000):
        """
        Summarize 'context' using Azure OpenAI chat completions, ensuring a string is returned.
        Significantly increased tokens for comprehensive insurance policy summaries with o3.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 보험약관 문서의 검색 최적화를 위한 전문적인 요약을 제공하는 어시스턴트입니다. "
                    "다음 원칙을 따르세요:\n"
                    "1. 정보 보존: 중요한 세부사항, 수치, 조건, 예외사항을 누락하지 마세요\n"
                    "2. 검색 최적화: 다양한 검색 키워드가 포함되도록 동의어, 관련 용어를 포함하세요\n"
                    "3. 계층적 표현: 여러 하위 내용을 포괄하는 상위 개념을 명확히 제시하세요\n"
                    "4. 구조적 정보: 보장 범위, 지급 조건, 제외 사항 등의 구조를 명확히 하세요\n"
                    "5. 맥락 유지: 약관의 법적 맥락과 실무적 의미를 보존하세요"
                ),
            },
            {
                "role": "user",
                "content": f"다음 보험약관 텍스트를 검색 최적화를 위해 포괄적으로 요약해 주세요. "
                f"핵심 정보를 누락하지 말고, 검색에 유용한 키워드와 개념을 충분히 포함하세요:\n\n{context}",
            },
        ]
        try:
            resp = self.client.chat(
                messages=messages, max_completion_tokens=max_tokens, temperature=0.0
            )

            # Attempt to extract text from response
            choice = resp.choices[0]

            # Primary path: choice.message.content
            content = (
                getattr(choice.message, "content", None)
                if hasattr(choice, "message")
                else None
            )

            # Fallback path: choice.text
            if not content and hasattr(choice, "text"):
                content = choice.text

            # Ensure we have a string
            if not isinstance(content, str):
                raise ValueError(
                    f"AzureSummarizationModel returned non-string content: {content!r}"
                )

            return content.strip()

        except Exception as e:
            logging.warning(f"Summarization failed due to: {e}")
            logging.warning(f"Filtered Content Sample: {context[:200]!r}")
            return "⚠️ Skipped summarization due to content filter trigger."
