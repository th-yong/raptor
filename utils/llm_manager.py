import os
from typing import List, Dict, Any
from openai import AzureOpenAI


class AzureAIClientManager:
    """
    Azure OpenAI Embedding & Chat API를 통합 관리하는 매니저.
    환경변수 또는 생성자 인자로 endpoint, key, deployment 설정을 주입할 수 있습니다.
    """

    ALLOWED_EMBEDDING_DEPLOYMENT = {"text-embedding-3-large"}
    ALLOWED_CHAT_DEPLOYMENTS = {
        "gpt-4o",  # model version 2024-11-20
    }

    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-12-01-preview",
        deployment: str = None,
    ):
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.deployment = deployment

        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        list[str] → list[vector]
        """
        if self.deployment not in self.ALLOWED_EMBEDDING_DEPLOYMENT:
            raise ValueError(
                f"임베딩은 `{self.ALLOWED_EMBEDDING_DEPLOYMENT}`에서만 가능합니다. "
                f"현재 deployment: {self.deployment!r}"
            )

        try:
            resp = self._client.embeddings.create(
                input=texts, model=self.deployment, **kwargs
            )
            # index 순 정렬
            items = sorted(resp.data, key=lambda d: d.index)
            return [it.embedding for it in items]
        except Exception as e:
            print(e)
            # 필요시 재시도/로깅 추가
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        OpenAI Chat Completions 호출.
        messages 포맷은 OpenAI Chat API 형태를 따릅니다.
        """
        if self.deployment not in self.ALLOWED_CHAT_DEPLOYMENTS:
            raise ValueError(
                f"Chat must use one of {self.ALLOWED_CHAT_DEPLOYMENTS}, current: {self.deployment!r}"
            )

        try:
            result = self._client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return result
        except Exception as e:
            print(e)
            raise
