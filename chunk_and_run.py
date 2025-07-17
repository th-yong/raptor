import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from raptor.ClusterSemanticTableTextSplitter import ClusterSemanticTableTextSplitter
from utils.llm_manager import AzureAIClientManager
from raptor.EmbeddingModels import AzureEmbeddingModel
from raptor.SummarizationModels import AzureSummarizationModel
from raptor.QAModels import AzureQAModel
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig


def create_azure_clients() -> Tuple[AzureEmbeddingModel, AzureSummarizationModel, AzureQAModel]:
    """
    Initialise Azure AI clients for embedding and chat completion models.

    Environment variables required
    ------------------------------
    AZURE_OPENAI_ENDPOINT : str
        The endpoint of your Azure Cognitive Services resource.
    AZURE_OPENAI_KEY : str
        The API key for the resource.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")

    if not endpoint or not api_key:
        raise EnvironmentError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set.")

    emb_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment="text-embedding-3-large",
    )
    chat_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment="gpt-4o",
    )

    return (
        AzureEmbeddingModel(emb_client),
        AzureSummarizationModel(chat_client),
        AzureQAModel(chat_client),
    )


def build_retrieval_augmentation(
    embedding_model: AzureEmbeddingModel,
    summarization_model: AzureSummarizationModel,
    qa_model: AzureQAModel,
) -> RetrievalAugmentation:
    """
    Create a RetrievalAugmentation instance with sensible defaults.
    """
    cfg = RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        summarization_model=summarization_model,
        qa_model=qa_model,
        tb_max_tokens=512,
        tb_summarization_length=512,
    )
    return RetrievalAugmentation(config=cfg)


def ingest_document(ra: RetrievalAugmentation, doc_path: Path) -> None:
    """
    Read a document, split it into semantic chunks, and register it with RAPTOR.
    """
    text = doc_path.read_text(encoding="utf-8")
    splitter = ClusterSemanticTableTextSplitter()
    chunks = splitter.split_documents([text])
    ra.add_documents(docs = None, chunked_list=chunks) # If you want to add chunked documents
    # ra.add_documents(docs = text, chunked_list=None)  # If you want to add raw text instead of chunks


def main() -> None:
    load_dotenv()  # Load variables from a .env file if present.

    emb_model, sum_model, qa_model = create_azure_clients()
    ra = build_retrieval_augmentation(emb_model, sum_model, qa_model)

    document = Path("demo/현대해상3(퇴직연금)상품약관.txt")
    ingest_document(ra, document)

    # Debug: inspect tree nodes
    tree = ra.tree
    # print("Tree Structure:", list(map(lambda x: x.text, tree.all_nodes.values())))

    # Example Q&A
    question = "이율적용형 이율 비율이란?"
    answer, _layers = ra.answer_question(question, return_layer_information=True)
    print(f"Q: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
