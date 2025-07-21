import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

from dotenv import load_dotenv
from raptor.ClusterSemanticTableTextSplitter import ClusterSemanticTableTextSplitter
from utils.llm_manager import AzureAIClientManager
from raptor.EmbeddingModels import AzureEmbeddingModel
from raptor.SummarizationModels import AzureSummarizationModel
from raptor.QAModels import AzureQAModel
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig


def create_azure_clients() -> (
    Tuple[AzureEmbeddingModel, AzureSummarizationModel, AzureQAModel]
):
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
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set."
        )

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


def load_and_group_csv_data(csv_path: str) -> Dict[str, List[str]]:
    """
    Load CSV file and group KO_content by source_title.

    Args:
        csv_path: Path to the character.csv file

    Returns:
        Dictionary with source_title as key and list of KO_content as value
    """
    print(f"Loading CSV data from {csv_path}...")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Group by source_title and collect KO_content
    grouped_data = defaultdict(list)

    for _, row in df.iterrows():
        source_title = row["source_title"]
        ko_content = row["KO_content"]

        # Skip empty or NaN content
        if pd.notna(ko_content) and ko_content.strip():
            grouped_data[source_title].append(ko_content.strip())

    print(f"Found {len(grouped_data)} unique source titles")
    for title, contents in grouped_data.items():
        print(f"  - {title}: {len(contents)} chunks")

    return dict(grouped_data)


def process_source_title(
    source_title: str,
    chunked_list: List[str],
    embedding_model: AzureEmbeddingModel,
    summarization_model: AzureSummarizationModel,
    qa_model: AzureQAModel,
    results_dir: Path,
    skip_existing: bool = True,
) -> bool:
    """
    Process a single source_title with its chunked content.

    Args:
        source_title: Name of the source document
        chunked_list: List of pre-chunked content
        embedding_model: Azure embedding model
        summarization_model: Azure summarization model
        qa_model: Azure QA model
        results_dir: Directory to save results
        skip_existing: Whether to skip if result file already exists

    Returns:
        bool: True if processed, False if skipped
    """
    # Create safe filename from source_title
    safe_filename = "".join(
        c for c in source_title if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    safe_filename = safe_filename.replace(" ", "_")
    save_path = results_dir / f"{safe_filename}.pkl"

    # Check if file already exists
    if skip_existing and save_path.exists():
        print(f"\n⏭️  Skipping: {source_title}")
        print(f"   Result already exists: {save_path}")
        return False

    print(f"\n🔄 Processing: {source_title}")
    print(f"   Number of chunks: {len(chunked_list)}")
    print(f"   Will save to: {save_path}")

    # Create RetrievalAugmentation instance
    ra = build_retrieval_augmentation(embedding_model, summarization_model, qa_model)

    # Add documents using the pre-chunked content
    ra.add_documents(docs=None, chunked_list=chunked_list)

    # Save the tree
    ra.save(str(save_path))
    print(f"✅ Saved to: {save_path}")
    return True


def main(skip_existing: bool = True) -> None:
    load_dotenv()  # Load variables from a .env file if present.

    # Create results directory
    results_dir = Path("results/recursive")
    results_dir.mkdir(exist_ok=True)

    # Load CSV data and group by source_title
    csv_path = "input/recursive.csv"
    grouped_data = load_and_group_csv_data(csv_path)

    # Check existing files if skip_existing is True
    if skip_existing:
        existing_files = list(results_dir.glob("*.pkl"))
        print(
            f"\n📁 Found {len(existing_files)} existing result files in {results_dir}"
        )
        if existing_files:
            print("   Existing files:")
            for file in existing_files[:5]:  # Show first 5
                print(f"   - {file.name}")
            if len(existing_files) > 5:
                print(f"   ... and {len(existing_files) - 5} more")

    # Create Azure clients once
    emb_model, sum_model, qa_model = create_azure_clients()

    # Process each source_title
    processed_count = 0
    skipped_count = 0
    error_count = 0

    total_sources = len(grouped_data)
    print(f"\n🚀 Starting processing of {total_sources} source titles...")

    for i, (source_title, chunked_list) in enumerate(grouped_data.items(), 1):
        try:
            print(f"\n[{i}/{total_sources}]", end="")
            was_processed = process_source_title(
                source_title=source_title,
                chunked_list=chunked_list,
                embedding_model=emb_model,
                summarization_model=sum_model,
                qa_model=qa_model,
                results_dir=results_dir,
                skip_existing=skip_existing,
            )

            if was_processed:
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            error_count += 1
            print(f"\n❌ Error processing {source_title}: {e}")
            continue

    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 Processing Summary:")
    print(f"   Total sources: {total_sources}")
    print(f"   ✅ Processed: {processed_count}")
    print(f"   ⏭️  Skipped (already exists): {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📁 Results saved in: {results_dir}")
    print(f"=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process character.csv and create RAPTOR trees for each source_title"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if result files already exist",
    )

    args = parser.parse_args()

    # If --force is specified, don't skip existing files
    skip_existing = not args.force

    main(skip_existing=skip_existing)
