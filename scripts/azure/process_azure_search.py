#!/usr/bin/env python3
"""
Azure AI Search 통합 문서 처리 스크립트:
1. Azure AI Search에서 데이터를 조회하여 source_title별로 그룹핑
2. 각 source_title별로 RAPTOR tree를 생성하고 PKL로 저장
3. 생성된 PKL 파일들을 읽어서 하나의 CSV 파일로 병합 (선택사항)
4. RAPTOR 결과를 새로운 Azure AI Search 인덱스에 업로드

사용 예시:
# AI Search에서 데이터 조회하여 RAPTOR 처리 후 새 인덱스에 업로드
uv run process_azure_search.py --source-index myindex --target-index myindex_raptor

# PKL 생성만 수행
uv run process_azure_search.py --source-index myindex --skip-upload

# 기존 PKL에서 새 인덱스로 업로드만 수행
uv run process_azure_search.py --target-index myindex_raptor --skip-processing

"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict
import traceback
import argparse
from datetime import datetime

from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

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
# Azure AI Search 연동 함수들
# ===============================


def create_search_client(index_name: str) -> Tuple[SearchClient, SearchIndexClient]:
    """Azure AI Search 클라이언트 생성"""
    endpoint = os.getenv("durable-azure-search-endpoint")
    api_key = os.getenv("durable-azure-search-api-key")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "durable-azure-search-endpoint and durable-azure-search-api-key must be set."
        )

    credential = AzureKeyCredential(api_key)
    search_client = SearchClient(
        endpoint=endpoint, index_name=index_name, credential=credential
    )
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    return search_client, index_client


def fetch_data_from_search(
    search_client: SearchClient, group_by_field: str = "source_title"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Azure AI Search에서 모든 데이터를 조회하고 group_by_field로 그룹핑

    Args:
        search_client: Azure Search 클라이언트
        group_by_field: 그룹핑할 필드명 (기본값: "source_title")

    Returns:
        Dictionary with group_by_field value as key and list of documents
    """
    print(f"🔍 Fetching data from Azure AI Search...")

    # 모든 문서 조회 (페이징 처리)
    all_documents = []
    skip = 0
    top = 1000  # 페이지 크기

    while True:
        try:
            results = search_client.search(
                search_text="*",
                select=["*"],
                skip=skip,
                top=top,
                include_total_count=True,
            )

            batch_documents = list(results)
            if not batch_documents:
                break

            all_documents.extend(batch_documents)
            skip += top

            print(f"   📄 Fetched {len(all_documents)} documents so far...")

            # 더 이상 문서가 없으면 중단
            if len(batch_documents) < top:
                break

        except Exception as e:
            print(f"❌ Error fetching documents: {e}")
            break

    print(f"✅ Total {len(all_documents)} documents fetched")

    # 그룹핑
    grouped_data = defaultdict(list)

    for doc in all_documents:
        group_key = doc.get(group_by_field, "unknown")
        # KO_content 필드명 확인 (대소문자 구분)
        ko_content = doc.get("KO_content", doc.get("ko_content", ""))
        ko_content_vector = doc.get(
            "KO_content_vector", doc.get("ko_content_vector", None)
        )

        # 빈 내용 건너뛰기
        if not ko_content or str(ko_content).strip() == "":
            continue

        doc_data = {
            "text": str(ko_content).strip(),
            "vector": ko_content_vector,
            "original_doc": doc,  # 원본 문서 정보 보존
        }
        grouped_data[group_key].append(doc_data)

    print(f"\n📊 Data grouped by '{group_by_field}':")
    for group_key, docs in grouped_data.items():
        print(f"   - {group_key}: {len(docs)} documents")

    return dict(grouped_data)


def create_raptor_index(
    index_client: SearchIndexClient, index_name: str, vector_dimensions: int = 3072
) -> bool:
    """
    RAPTOR 결과를 저장할 새로운 인덱스 생성

    Args:
        index_client: Azure Search Index 클라이언트
        index_name: 생성할 인덱스 이름
        vector_dimensions: 벡터 차원 수 (기본값: 3072)
    """
    print(f"🏗️  Creating new index: {index_name}")

    try:
        # 기존 인덱스가 있는지 확인
        try:
            existing_index = index_client.get_index(index_name)
            print(f"⚠️  Index '{index_name}' already exists. Skipping creation.")
            return True
        except:
            pass  # 인덱스가 없으면 새로 생성

        # 벡터 검색 설정
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-config",
                    algorithm_configuration_name="my-algorithms-config",
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
        )

        # 필드 정의
        fields = [
            SearchField(
                name="id", type=SearchFieldDataType.String, key=True, searchable=False
            ),
            SearchField(
                name="source_title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchField(
                name="node_index",
                type=SearchFieldDataType.Int32,
                searchable=False,
                filterable=True,
            ),
            SearchField(
                name="ko_content", type=SearchFieldDataType.String, searchable=True
            ),
            SearchField(
                name="ko_content_vector",
                type=SearchFieldDataType.Collection("Edm.Half"),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="my-vector-config",
            ),
            SearchField(
                name="children", type=SearchFieldDataType.String, searchable=False
            ),
            SearchField(
                name="layer",
                type=SearchFieldDataType.Int32,
                searchable=False,
                filterable=True,
            ),
            SearchField(
                name="is_root",
                type=SearchFieldDataType.Boolean,
                searchable=False,
                filterable=True,
            ),
            SearchField(
                name="is_leaf",
                type=SearchFieldDataType.Boolean,
                searchable=False,
                filterable=True,
            ),
            SearchField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                searchable=False,
                filterable=True,
            ),
        ]

        # 인덱스 생성
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

        result = index_client.create_index(index)
        print(f"✅ Index '{index_name}' created successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to create index '{index_name}': {e}")
        return False


def upload_raptor_results_to_search(
    search_client: SearchClient, all_raptor_data: List[Dict[str, Any]]
) -> bool:
    """
    RAPTOR 결과를 Azure AI Search에 업로드

    Args:
        search_client: Azure Search 클라이언트
        all_raptor_data: RAPTOR 처리 결과 데이터
    """
    print(f"⬆️  Uploading {len(all_raptor_data)} RAPTOR results to Azure AI Search...")

    try:
        import base64
        import hashlib

        # 배치 단위로 업로드 (Azure Search는 1000개씩 제한)
        batch_size = 100
        current_time = datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )  # ISO 형식으로 변경

        for i in range(0, len(all_raptor_data), batch_size):
            batch = all_raptor_data[i : i + batch_size]

            # 각 문서에 고유 ID와 타임스탬프 추가
            documents = []
            for j, item in enumerate(batch):
                # 안전한 문서 키 생성 (Base64 인코딩)
                original_id = f"{item['source_title']}_{item['node_index']}_{i+j}"
                safe_id = (
                    base64.urlsafe_b64encode(original_id.encode("utf-8"))
                    .decode("ascii")
                    .rstrip("=")
                )

                # 너무 긴 ID 방지 (해시 사용)
                if len(safe_id) > 100:  # Azure Search ID 길이 제한
                    hash_id = hashlib.md5(original_id.encode("utf-8")).hexdigest()
                    safe_id = f"raptor_{hash_id}"

                doc = {
                    "id": safe_id,
                    "source_title": item["source_title"],
                    "node_index": item["node_index"],
                    "ko_content": item["ko_content"],
                    "ko_content_vector": (
                        json.loads(item["ko_content_vector"])
                        if item["ko_content_vector"] != "[]"
                        else []
                    ),
                    "children": item["children"],
                    "layer": item["layer"],
                    "is_root": item["is_root"],
                    "is_leaf": item["is_leaf"],
                    "created_at": current_time,
                }
                documents.append(doc)

            # 업로드
            result = search_client.upload_documents(documents)

            # 결과 확인
            success_count = sum(1 for r in result if r.succeeded)
            print(
                f"   📤 Batch {i//batch_size + 1}: {success_count}/{len(batch)} documents uploaded"
            )

            if success_count != len(batch):
                failed_docs = [r for r in result if not r.succeeded]
                for failed in failed_docs:
                    print(f"      ❌ Failed: {failed.key} - {failed.error_message}")

        print(f"✅ Upload completed!")
        return True

    except Exception as e:
        print(f"❌ Failed to upload to Azure AI Search: {e}")
        traceback.print_exc()
        return False


# ===============================
# RAPTOR 처리 함수들 (기존 코드 재사용)
# ===============================


def create_azure_clients() -> (
    Tuple[AzureEmbeddingModel, AzureSummarizationModel, AzureQAModel]
):
    """Azure AI clients for embedding and chat completion models."""
    endpoint = os.getenv("durable-azure-openai-endpoint")
    api_key = os.getenv("durable-azure-openai-key")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "durable-azure-openai-endpoint and durable-azure-openai-key must be set."
        )

    # 임베딩용 클라이언트 (text-embedding-3-large deployment 사용)
    emb_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment=os.getenv("durable-azure-openai-embedding-deployment"),
    )

    # 채팅용 클라이언트 (o3 deployment 사용)
    chat_client = AzureAIClientManager(
        endpoint=endpoint,
        api_key=api_key,
        deployment=os.getenv("durable-azure-openai-deployment"),
    )

    print(
        f"🔧 임베딩 deployment: {os.getenv('durable-azure-openai-embedding-deployment')}"
    )
    print(f"🔧 채팅 deployment: {os.getenv('durable-azure-openai-deployment')}")

    # 생성된 클라이언트들의 deployment 확인
    emb_model = AzureEmbeddingModel(emb_client)
    sum_model = AzureSummarizationModel(chat_client)
    qa_model = AzureQAModel(chat_client)

    print(f"🔧 임베딩 모델 deployment: {emb_client.deployment}")
    print(f"🔧 채팅 모델 deployment: {chat_client.deployment}")

    return (emb_model, sum_model, qa_model)


def build_retrieval_augmentation(
    embedding_model: AzureEmbeddingModel,
    summarization_model: AzureSummarizationModel,
    qa_model: AzureQAModel,
) -> RetrievalAugmentation:
    """Create a RetrievalAugmentation instance with sensible defaults."""
    cfg = RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        summarization_model=summarization_model,
        qa_model=qa_model,
        tb_max_tokens=2000,  # o3 최적화: 대폭 확장
        tb_summarization_length=4000,  # o3 최적화: 대폭 확장
    )
    return RetrievalAugmentation(config=cfg)


def process_source_title_from_search(
    source_title: str,
    documents: List[Dict[str, Any]],
    embedding_model: AzureEmbeddingModel,
    summarization_model: AzureSummarizationModel,
    qa_model: AzureQAModel,
    results_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Optional[Any]:
    """
    Process a single source_title with its documents from Azure Search.

    Returns:
        RetrievalAugmentation object if processed, None if skipped
    """
    if results_dir:
        save_path = results_dir / f"{source_title}.pkl"

        # Check if file already exists
        if skip_existing and save_path.exists():
            print(f"\n⏭️  Skipping: {source_title}")
            print(f"   Result already exists: {save_path}")
            return None

    print(f"\n🔄 Processing: {source_title}")
    print(f"   Number of documents: {len(documents)}")
    if results_dir:
        print(f"   Will save to: {save_path}")

    # Create RetrievalAugmentation instance
    ra = build_retrieval_augmentation(embedding_model, summarization_model, qa_model)

    # Add documents using the content from Azure Search
    chunked_list = [doc["text"] for doc in documents]
    vectors = [doc.get("vector") for doc in documents]

    ra.add_documents(
        docs=None,
        chunked_list=chunked_list,
        vectors=vectors,
    )

    # Save the tree if results_dir is provided
    if results_dir:
        ra.save(str(save_path))
        print(f"✅ Saved to: {save_path}")

    return ra


# ===============================
# PKL 파일 병합 함수들 (기존 코드 재사용)
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
    """RAPTOR Tree 객체에서 CSV 형식에 맞는 정보 추출"""
    rows = []

    if hasattr(data, "all_nodes") and hasattr(data, "layer_to_nodes"):
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


def extract_node_info(node, source_title, node_to_layer):
    """개별 Node 객체에서 정보 추출"""
    try:
        content = getattr(node, "text", "")
        node_index = getattr(node, "index", 0)
        children = getattr(node, "children", set())

        embeddings_dict = getattr(node, "embeddings", {})
        vector = embeddings_dict.get("EMB", []) if embeddings_dict else []

        layer = node_to_layer.get(node_index, 0)

        is_leaf = len(children) == 0
        is_root = layer == max(node_to_layer.values()) if node_to_layer else False

        # 벡터를 문자열로 변환
        if isinstance(vector, (list, tuple, np.ndarray)):
            if len(vector) > 0:
                vector_str = json.dumps(
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
            else:
                vector_str = "[]"
        else:
            vector_str = "[]"

        # children을 문자열로 변환
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


# ===============================
# MAIN 함수
# ===============================


def main(
    source_index: Optional[str] = None,
    target_index: Optional[str] = None,
    group_by_field: str = "source_title",
    results_dir: Optional[str] = None,
    skip_existing: bool = True,
    skip_processing: bool = False,
    skip_upload: bool = False,
    save_csv: bool = False,
) -> None:
    """
    메인 처리 함수

    Args:
        source_index: 소스 Azure AI Search 인덱스 이름
        target_index: 타겟 Azure AI Search 인덱스 이름
        group_by_field: 데이터 그룹핑 필드
        results_dir: PKL 파일 저장 디렉토리
        skip_existing: 기존 PKL 파일 건너뛰기
        skip_processing: RAPTOR 처리 건너뛰기
        skip_upload: Azure Search 업로드 건너뛰기
        save_csv: CSV 파일 저장 여부
    """

    load_dotenv()

    print("=" * 80)
    print(f"🚀 Azure AI Search 통합 문서 처리 시작")
    print(f"   Source Index: {source_index}")
    print(f"   Target Index: {target_index}")
    print(f"   Group By Field: {group_by_field}")
    print(f"   Results Directory: {results_dir}")
    print(f"   Skip Existing: {skip_existing}")
    print(f"   Skip Processing: {skip_processing}")
    print(f"   Skip Upload: {skip_upload}")
    print(f"   Save CSV: {save_csv}")
    print("=" * 80)

    # Results directory 설정
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
    else:
        results_path = None

    all_raptor_data = []

    # STEP 1: Azure AI Search에서 데이터 조회 및 RAPTOR 처리
    if not skip_processing:
        if not source_index:
            print("❌ Source index must be provided for processing")
            return

        print(f"\n🔍 STEP 1: Azure AI Search 데이터 조회 및 RAPTOR 처리")
        print("-" * 50)

        try:
            # Azure Search 클라이언트 생성
            search_client, _ = create_search_client(source_index)

            # 데이터 조회 및 그룹핑
            grouped_data = fetch_data_from_search(search_client, group_by_field)

            if not grouped_data:
                print("❌ No data found in the source index")
                return

            # Azure AI 클라이언트 생성
            emb_model, sum_model, qa_model = create_azure_clients()

            # 각 그룹별 RAPTOR 처리
            processed_count = 0
            skipped_count = 0
            error_count = 0

            total_groups = len(grouped_data)
            print(f"\n🚀 Starting processing of {total_groups} groups...")

            for i, (group_key, documents) in enumerate(grouped_data.items(), 1):
                try:
                    print(f"\n[{i}/{total_groups}]", end="")

                    ra_result = process_source_title_from_search(
                        source_title=group_key,
                        documents=documents,
                        embedding_model=emb_model,
                        summarization_model=sum_model,
                        qa_model=qa_model,
                        results_dir=results_path,
                        skip_existing=skip_existing,
                    )

                    if ra_result is not None:
                        processed_count += 1

                        # RAPTOR 결과 추출
                        raptor_rows = extract_raptor_data(ra_result, group_key)
                        all_raptor_data.extend(raptor_rows)
                        print(f"   ✅ Extracted {len(raptor_rows)} RAPTOR nodes")
                    else:
                        skipped_count += 1

                        # 기존 PKL 파일에서 데이터 로드
                        if results_path:
                            pkl_path = results_path / f"{group_key}.pkl"
                            if pkl_path.exists():
                                pkl_data = load_pkl_file(pkl_path)
                                if pkl_data:
                                    raptor_rows = extract_raptor_data(
                                        pkl_data, group_key
                                    )
                                    all_raptor_data.extend(raptor_rows)
                                    print(
                                        f"   📁 Loaded {len(raptor_rows)} nodes from existing PKL"
                                    )

                except Exception as e:
                    error_count += 1
                    print(f"\n❌ Error processing {group_key}: {e}")
                    traceback.print_exc()
                    continue

            # Processing Summary
            print(f"\n" + "-" * 60)
            print(f"📊 RAPTOR 처리 결과:")
            print(f"   Total groups: {total_groups}")
            print(f"   ✅ Processed: {processed_count}")
            print(f"   ⏭️  Skipped (already exists): {skipped_count}")
            print(f"   ❌ Errors: {error_count}")
            print(f"   📄 Total RAPTOR nodes: {len(all_raptor_data)}")
            print(f"-" * 60)

        except Exception as e:
            print(f"❌ Error in processing step: {e}")
            traceback.print_exc()
            return

    else:
        print(f"\n⏭️  STEP 1 SKIPPED: RAPTOR 처리 건너뜀")

        # 기존 PKL 파일들에서 데이터 로드
        if results_path and results_path.exists():
            pkl_files = list(results_path.glob("*.pkl"))
            print(f"📁 Loading data from {len(pkl_files)} existing PKL files...")

            for pkl_file in pkl_files:
                source_title = pkl_file.stem
                pkl_data = load_pkl_file(pkl_file)
                if pkl_data:
                    raptor_rows = extract_raptor_data(pkl_data, source_title)
                    all_raptor_data.extend(raptor_rows)
                    print(f"   📁 Loaded {len(raptor_rows)} nodes from {pkl_file.name}")

    # STEP 2: CSV 저장 (선택사항)
    if save_csv and all_raptor_data:
        print(f"\n📄 STEP 2: CSV 파일 저장")
        print("-" * 50)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"raptor_results_{timestamp}.csv"

        df = pd.DataFrame(all_raptor_data)
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(df)} rows to {csv_filename}")

    # STEP 3: Azure AI Search 업로드
    if not skip_upload and all_raptor_data:
        if not target_index:
            print("❌ Target index must be provided for upload")
            return

        print(f"\n⬆️  STEP 3: Azure AI Search 업로드")
        print("-" * 50)

        try:
            # 타겟 인덱스 클라이언트 생성
            target_search_client, index_client = create_search_client(target_index)

            # 타겟 인덱스 생성 (존재하지 않는 경우)
            create_raptor_index(index_client, target_index)

            # 데이터 업로드
            upload_raptor_results_to_search(target_search_client, all_raptor_data)

        except Exception as e:
            print(f"❌ Error in upload step: {e}")
            traceback.print_exc()
            return
    else:
        if skip_upload:
            print(f"\n⏭️  STEP 3 SKIPPED: Azure AI Search 업로드 건너뜀")
        elif not all_raptor_data:
            print(f"\n⏭️  STEP 3 SKIPPED: 업로드할 데이터가 없음")

    # Final Summary
    print(f"\n" + "=" * 80)
    print(f"🏁 전체 작업 완료!")
    if results_path:
        print(f"   Results Directory: {results_path}")
    if all_raptor_data:
        print(f"   Total RAPTOR nodes processed: {len(all_raptor_data)}")
    if not skip_upload and target_index:
        print(f"   Target Index: {target_index}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Azure AI Search 통합 문서 처리: Search → RAPTOR → Search"
    )
    parser.add_argument(
        "--source-index", type=str, help="소스 Azure AI Search 인덱스 이름"
    )
    parser.add_argument(
        "--target-index", type=str, help="타겟 Azure AI Search 인덱스 이름"
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="source_title",
        help="데이터 그룹핑 필드명 (default: source_title)",
    )
    parser.add_argument(
        "--results-dir", type=str, help="PKL 파일 저장 디렉토리 (선택사항)"
    )
    parser.add_argument(
        "--force", action="store_true", help="기존 PKL 파일이 있어도 강제로 재처리"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="RAPTOR 처리 건너뛰기 (업로드만 수행)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Azure Search 업로드 건너뛰기 (RAPTOR 처리만 수행)",
    )
    parser.add_argument(
        "--save-csv", action="store_true", help="결과를 CSV 파일로도 저장"
    )

    args = parser.parse_args()

    # Validation
    if not args.skip_processing and not args.source_index:
        print("❌ --source-index is required when not skipping processing")
        exit(1)

    if not args.skip_upload and not args.target_index:
        print("❌ --target-index is required when not skipping upload")
        exit(1)

    # If --force is specified, don't skip existing files
    skip_existing = not args.force

    main(
        source_index=args.source_index,
        target_index=args.target_index,
        group_by_field=args.group_by,
        results_dir=args.results_dir,
        skip_existing=skip_existing,
        skip_processing=args.skip_processing,
        skip_upload=args.skip_upload,
        save_csv=args.save_csv,
    )
