#!/usr/bin/env python3
"""
통합 문서 처리 스크립트:
1. CSV 파일을 읽어서 각 source_title별로 RAPTOR tree를 생성하고 PKL로 저장
2. 생성된 PKL 파일들을 읽어서 하나의 CSV 파일로 병합

# 기본 실행 (기존 파일 건너뛰기)
uv run process_and_merge_documents.py

# 강제 재처리 (기존 PKL 파일 덮어쓰기)
uv run process_and_merge_documents.py --force

# PKL 생성만 수행
uv run process_and_merge_documents.py --skip-merging

# 병합만 수행 (이미 PKL 파일이 있을 때)
uv run process_and_merge_documents.py --skip-processing

# 다른 데이터셋 처리
uv run process_and_merge_documents.py --terms eterms --chunk table

"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Dict, Any
from collections import defaultdict
import traceback
import argparse

from dotenv import load_dotenv
from raptor.ClusterSemanticTableTextSplitter import ClusterSemanticTableTextSplitter
from utils.llm_manager import AzureAIClientManager
from raptor.EmbeddingModels import AzureEmbeddingModel
from raptor.SummarizationModels import AzureSummarizationModel
from raptor.QAModels import AzureQAModel
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

# Suppress FutureWarnings globally
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ===============================
# PART 1: 문서 처리 및 PKL 생성
# ===============================


def create_azure_clients() -> (
    Tuple[AzureEmbeddingModel, AzureSummarizationModel, AzureQAModel]
):
    """
    Initialise Azure AI clients for embedding and chat completion models.

    Environment variables required
    ------------------------------
    AZURE_OPENAI_ENDPOINT : str
        The endpoint of your Azure Cognitive Services resource.
    AZURE_OPENAI_API_KEY : str
        The API key for the resource.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
        )

    emb_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
    )
    chat_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment=os.getenv("AZURE_COMPLETION_DEPLOYMENT_NAME"),
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
        tb_max_tokens=2000,  # o3 최적화: 대폭 확장
        tb_summarization_length=4000,  # o3 최적화: 대폭 확장
    )
    return RetrievalAugmentation(config=cfg)


def load_and_group_csv_data(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CSV file and group KO_content by source_title.

    Args:
        csv_path: Path to the character.csv file

    Returns:
        Dictionary with source_title as key and list of dicts with text and optional vector
    """
    print(f"Loading CSV data from {csv_path}...")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Group by source_title and collect KO_content and optional KO_content_vector
    grouped_data = defaultdict(list)

    for _, row in df.iterrows():
        source_title = row["source_title"]
        ko_content = row["KO_content"]
        ko_content = "" if pd.isna(ko_content) else str(ko_content)
        vector = None
        if "KO_content_vector" in row and pd.notna(row["KO_content_vector"]):
            try:
                import ast

                vector = ast.literal_eval(row["KO_content_vector"])
                if isinstance(vector, list) and len(vector) != 3072:
                    print(
                        f"⚠️ Vector length mismatch for '{source_title}': expected 3072, got {len(vector)}. Ignoring vector."
                    )
                    vector = None
            except Exception as e:
                print(f"⚠️ Failed to parse vector for '{source_title}': {e}")
        grouped_data[source_title].append(
            {"text": ko_content.strip(), "vector": vector}
        )

    print(f"Found {len(grouped_data)} unique source titles")
    for title, contents in grouped_data.items():
        print(f"  - {title}: {len(contents)} chunks")

    return dict(grouped_data)


def process_source_title(
    source_title: str,
    chunked_list: List[Dict[str, Any]],
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
        chunked_list: List of dicts with 'text' and optional 'vector'
        embedding_model: Azure embedding model
        summarization_model: Azure summarization model
        qa_model: Azure QA model
        results_dir: Directory to save results
        skip_existing: Whether to skip if result file already exists

    Returns:
        bool: True if processed, False if skipped
    """
    save_path = results_dir / f"{source_title}.pkl"

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
    ra.add_documents(
        docs=None,
        chunked_list=[chunk["text"] for chunk in chunked_list],
        vectors=[chunk.get("vector") for chunk in chunked_list],
    )

    # Save the tree
    ra.save(str(save_path))
    print(f"✅ Saved to: {save_path}")
    return True


# ===============================
# PART 2: PKL 파일 병합 및 CSV 생성
# ===============================


def load_pkl_file(pkl_path):
    """pkl 파일을 로드하고 데이터 구조를 파악"""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def extract_raptor_data(data, source_title):
    """
    RAPTOR Tree 객체에서 CSV 형식에 맞는 정보 추출
    Tree 객체는 all_nodes, layer_to_nodes 등의 속성을 가짐
    """
    rows = []

    # Tree 객체인지 확인
    if hasattr(data, "all_nodes") and hasattr(data, "layer_to_nodes"):
        # all_nodes에서 모든 노드 정보 추출
        all_nodes = data.all_nodes
        layer_to_nodes = data.layer_to_nodes

        # 각 노드의 레이어 정보를 매핑
        node_to_layer = {}
        for layer_str, nodes_list in layer_to_nodes.items():
            layer_num = int(layer_str)
            for node in nodes_list:
                node_to_layer[node.index] = layer_num

        # 모든 노드 처리
        for node_id, node in all_nodes.items():
            row = extract_node_info(node, source_title, node_to_layer)
            if row:
                rows.append(row)

    else:
        print(f"Unexpected data structure for {source_title} - not a Tree object")
        return rows

    return rows


def extract_node_info(node, source_title, node_to_layer):
    """개별 Node 객체에서 정보 추출"""
    try:
        # Node 객체에서 정보 추출
        content = getattr(node, "text", "")
        node_index = getattr(node, "index", 0)
        children = getattr(node, "children", set())

        # 임베딩 벡터 추출 (embeddings 딕셔너리에서 'EMB' 키)
        embeddings_dict = getattr(node, "embeddings", {})
        vector = embeddings_dict.get("EMB", []) if embeddings_dict else []

        # 레이어 정보
        layer = node_to_layer.get(node_index, 0)

        # 루트/리프 노드 판별
        is_leaf = len(children) == 0
        is_root = layer == max(node_to_layer.values()) if node_to_layer else False

        # 벡터를 문자열로 변환 (CSV 저장용)
        if isinstance(vector, (list, tuple, np.ndarray)):
            if len(vector) > 0:
                vector_str = json.dumps(
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
            else:
                vector_str = "[]"
        else:
            vector_str = "[]"

        # children을 문자열로 변환 (set을 list로 변환)
        if isinstance(children, set):
            children_list = list(children)
        elif isinstance(children, (list, tuple)):
            children_list = list(children)
        else:
            children_list = []

        children_str = json.dumps(children_list)

        return {
            "source_title": source_title,
            "node_index": node_index,
            "ko_content": str(content),
            "ko_content_vector": vector_str,
            "children": children_str,
            "layer": int(layer),
            "is_root": bool(is_root),
            "is_leaf": bool(is_leaf),
        }

    except Exception as e:
        print(
            f"Error extracting node info from node {getattr(node, 'index', 'unknown')}: {e}"
        )
        return None


def merge_pkl_to_csv(results_dir: Path, output_file: str) -> bool:
    """PKL 파일들을 읽어서 하나의 CSV 파일로 병합"""

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist!")
        return False

    # 모든 pkl 파일 찾기
    pkl_files = list(results_dir.glob("*.pkl"))
    print(f"\n📁 Found {len(pkl_files)} pkl files for merging")

    if not pkl_files:
        print("No pkl files found to merge!")
        return False

    all_rows = []

    # 각 pkl 파일 처리
    for pkl_file in pkl_files:
        print(f"🔄 Merging {pkl_file.name}...")

        # 파일명에서 source_title 추출 (확장자 제거)
        source_title = pkl_file.stem

        # pkl 파일 로드
        data = load_pkl_file(pkl_file)
        if data is None:
            continue

        # 데이터 추출
        rows = extract_raptor_data(data, source_title)
        all_rows.extend(rows)
        print(f"  ✅ Extracted {len(rows)} rows")

    if not all_rows:
        print("No data extracted from pkl files!")
        return False

    # DataFrame 생성
    df = pd.DataFrame(all_rows)

    # 컬럼 순서 맞추기
    columns = [
        "source_title",
        "node_index",
        "ko_content",
        "ko_content_vector",
        "children",
        "layer",
        "is_root",
        "is_leaf",
    ]
    df = df[columns]

    # CSV 저장
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved {len(df)} rows to {output_file}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")

    # 샘플 데이터 출력
    print("\n📊 Sample data:")
    print(df.head(3))

    return True


# ===============================
# MAIN 함수
# ===============================


def main(
    terms: str = "iterms",
    chunk: str = "basic",
    skip_existing: bool = True,
    skip_processing: bool = False,
    skip_merging: bool = False,
) -> None:
    """
    메인 처리 함수

    Args:
        terms: "eterms" or "iterms"
        chunk: "basic" or "table"
        skip_existing: 기존 PKL 파일이 있으면 건너뛸지 여부
        skip_processing: PKL 생성 과정을 건너뛸지 여부
        skip_merging: CSV 병합 과정을 건너뛸지 여부
    """

    load_dotenv()  # Load variables from a .env file if present.

    # Create results directory
    results_dir = Path(f"results/{terms}/{chunk}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일 경로
    csv_output_file = f"results/{terms}_{chunk}_combined_raptor_data.csv"

    print("=" * 80)
    print(f"🚀 통합 문서 처리 시작")
    print(f"   Terms: {terms}")
    print(f"   Chunk: {chunk}")
    print(f"   Results Directory: {results_dir}")
    print(f"   Output CSV: {csv_output_file}")
    print(f"   Skip Existing: {skip_existing}")
    print(f"   Skip Processing: {skip_processing}")
    print(f"   Skip Merging: {skip_merging}")
    print("=" * 80)

    # STEP 1: PKL 파일 생성
    if not skip_processing:
        print(f"\n📝 STEP 1: CSV 데이터 처리 및 PKL 파일 생성")
        print("-" * 50)

        # Load CSV data and group by source_title
        csv_path = f"input/{terms}_{chunk}.csv"

        if not Path(csv_path).exists():
            print(f"❌ Input CSV file not found: {csv_path}")
            return

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
        try:
            emb_model, sum_model, qa_model = create_azure_clients()
        except Exception as e:
            print(f"❌ Failed to create Azure clients: {e}")
            return

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
                traceback.print_exc()
                continue

        # Processing Summary
        print(f"\n" + "-" * 60)
        print(f"📊 PKL 생성 결과:")
        print(f"   Total sources: {total_sources}")
        print(f"   ✅ Processed: {processed_count}")
        print(f"   ⏭️  Skipped (already exists): {skipped_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"-" * 60)
    else:
        print(f"\n⏭️  STEP 1 SKIPPED: PKL 파일 생성 건너뜀")

    # STEP 2: PKL 파일들을 CSV로 병합
    if not skip_merging:
        print(f"\n📊 STEP 2: PKL 파일들을 CSV로 병합")
        print("-" * 50)

        success = merge_pkl_to_csv(results_dir, csv_output_file)

        if success:
            print(f"\n✅ 병합 완료!")
        else:
            print(f"\n❌ 병합 실패!")
    else:
        print(f"\n⏭️  STEP 2 SKIPPED: CSV 병합 건너뜀")

    # Final Summary
    print(f"\n" + "=" * 80)
    print(f"🏁 전체 작업 완료!")
    print(f"   Results Directory: {results_dir}")
    if not skip_merging:
        print(f"   Final CSV Output: {csv_output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="통합 문서 처리: CSV → PKL → 병합된 CSV"
    )
    parser.add_argument(
        "--terms",
        choices=["eterms", "iterms"],
        default="iterms",
        help="처리할 용어 집합 (default: iterms)",
    )
    parser.add_argument(
        "--chunk",
        choices=["basic", "table"],
        default="basic",
        help="청크 유형 (default: basic)",
    )
    parser.add_argument(
        "--force", action="store_true", help="기존 PKL 파일이 있어도 강제로 재처리"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="PKL 생성 과정 건너뛰기 (병합만 수행)",
    )
    parser.add_argument(
        "--skip-merging",
        action="store_true",
        help="CSV 병합 과정 건너뛰기 (PKL 생성만 수행)",
    )

    args = parser.parse_args()

    # If --force is specified, don't skip existing files
    skip_existing = not args.force

    main(
        terms=args.terms,
        chunk=args.chunk,
        skip_existing=skip_existing,
        skip_processing=args.skip_processing,
        skip_merging=args.skip_merging,
    )
