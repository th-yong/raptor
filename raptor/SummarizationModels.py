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
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
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
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
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
    def summarize(self, context: str, max_tokens: int = 500):
        """
        Summarize 'context' using Azure OpenAI chat completions, ensuring a string is returned.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 보험약관에 대해 안전한 요약을 제공하는 유용한 어시스턴트입니다."
                    "해당 약관에는 신체적, 정신적인 피해에 대한 보상을 논의할 수 있습니다."
                    "청킹되어 일부 내용이 누락될 수 있습니다."
                    "공식문서이므로 노골적·부적절한 내용거의 없겠지만, 포함되어 있으면 중립적인 표현으로 바꿔주세요."
                ),
            },
            {
                "role": "user",
                "content": f"다음 텍스트를 간결하게 요약해 주세요:\n\n{context}",
            },
        ]
        try:
            resp = self.client.chat(
                messages=messages, max_tokens=max_tokens, temperature=0.0
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
