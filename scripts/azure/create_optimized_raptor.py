#!/usr/bin/env python3
"""
원본 약관 데이터만을 리프 노드로 하는 최적화된 RAPTOR 트리 생성 스크립트
"""

import os
import sys
import argparse
import base64
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

from config.unified_index_schema import create_unified_index_config


def create_azure_clients():
    """Azure Search 클라이언트 생성"""
    load_dotenv()

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")

    if not endpoint or not api_key:
        raise ValueError("Azure Search endpoint and API key must be set in .env file")

    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    return index_client, endpoint, credential


def fetch_all_documents(search_client: SearchClient) -> List[Dict[str, Any]]:
    """인덱스에서 모든 문서 조회"""
    print(f"🔍 Fetching all documents from index...")

    all_documents = []
    skip = 0
    top = 1000

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

            if len(batch_documents) < top:
                break

        except Exception as e:
            print(f"❌ Error fetching documents: {e}")
            break

    print(f"✅ Total documents fetched: {len(all_documents)}")
    return all_documents


def get_max_node_index(raptor_documents: List[Dict[str, Any]]) -> int:
    """RAPTOR 문서들에서 최대 node_index 찾기"""
    max_index = 0
    for doc in raptor_documents:
        if "node_index" in doc and doc["node_index"] is not None:
            max_index = max(max_index, doc["node_index"])
    return max_index


def convert_terms_to_raptor_leaf(
    doc: Dict[str, Any], node_index: int
) -> Dict[str, Any]:
    """
    약관 문서를 RAPTOR 리프 노드로 변환

    Args:
        doc: 원본 약관 문서
        node_index: 할당할 새로운 노드 인덱스

    Returns:
        RAPTOR 리프 노드로 변환된 문서
    """
    raptor_node = {}

    # 공통 필드들 복사
    if "id" in doc:
        raptor_node["id"] = doc["id"]
    if "source_title" in doc:
        raptor_node["source_title"] = doc["source_title"]
    if "KO_content" in doc:
        raptor_node["KO_content"] = doc["KO_content"]
    if "KO_content_vector" in doc:
        raptor_node["KO_content_vector"] = doc["KO_content_vector"]

    # 모든 문서가 RAPTOR 노드로 통일
    raptor_node["doc_type"] = "raptor_node"

    # 약관 전용 필드들 유지 (원본 정보)
    if "page_number" in doc:
        raptor_node["page_number"] = doc["page_number"]
    if "info" in doc:
        raptor_node["info"] = doc["info"]

    # RAPTOR 리프 노드 설정
    raptor_node["node_index"] = node_index
    raptor_node["layer"] = 0  # 리프 노드
    raptor_node["is_root"] = False
    raptor_node["is_leaf"] = True  # 원본 약관 데이터는 리프 노드
    raptor_node["children"] = "[]"  # 빈 자식 리스트

    # 생성 시간 설정
    if "info" in doc and doc["info"] and "create_date" in doc["info"]:
        raptor_node["created_at"] = doc["info"]["create_date"]
    else:
        raptor_node["created_at"] = datetime.now(timezone.utc).isoformat()

    return raptor_node


def normalize_raptor_summary_node(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAPTOR 요약 노드만 정규화 (리프 노드는 제외)

    Args:
        doc: 원본 RAPTOR 문서

    Returns:
        정규화된 RAPTOR 요약 노드 (layer > 0인 것만)
    """
    # 리프 노드는 제외 (layer 0이거나 is_leaf가 True인 경우)
    if (doc.get("layer") == 0) or (doc.get("is_leaf") is True):
        return None

    normalized = {}

    # 공통 필드들 복사
    if "source_title" in doc:
        normalized["source_title"] = doc["source_title"]
    if "KO_content" in doc:
        normalized["KO_content"] = doc["KO_content"]
    if "KO_content_vector" in doc:
        normalized["KO_content_vector"] = doc["KO_content_vector"]

    # 모든 문서가 RAPTOR 노드로 통일
    normalized["doc_type"] = "raptor_node"

    # 요약 노드들은 원본 페이지 정보 없음
    normalized["page_number"] = None
    normalized["info"] = None

    # RAPTOR 필드들 복사
    if "node_index" in doc:
        normalized["node_index"] = doc["node_index"]
    if "layer" in doc:
        normalized["layer"] = doc["layer"]
    if "is_root" in doc:
        normalized["is_root"] = doc["is_root"]
    if "is_leaf" in doc:
        normalized["is_leaf"] = doc["is_leaf"]
    if "children" in doc:
        # children 필드는 문자열로 변환
        if isinstance(doc["children"], list):
            normalized["children"] = str(doc["children"])
        else:
            normalized["children"] = doc["children"]
    if "created_at" in doc:
        normalized["created_at"] = doc["created_at"]

    # RAPTOR ID를 Base64로 인코딩하여 안전하게 만들기
    if "id" in doc:
        original_id = str(doc["id"])
        if not original_id.startswith("raptor_"):
            original_id = f"raptor_{original_id}"
        safe_id = base64.b64encode(original_id.encode()).decode().rstrip("=")
        normalized["id"] = safe_id

    return normalized


def create_unified_index(
    index_client: SearchIndexClient, target_index: str, dim: int = 3072
) -> bool:
    """통합 인덱스 생성"""
    try:
        print(f"🔧 Creating optimized RAPTOR index: {target_index}")

        # 기존 인덱스가 있으면 삭제
        try:
            index_client.get_index(target_index)
            print(f"   ⚠️  Index {target_index} already exists. Deleting...")
            index_client.delete_index(target_index)
            print(f"   🗑️  Deleted existing index")
        except ResourceNotFoundError:
            print(f"   ✅ Index {target_index} does not exist, creating new one")

        # 새 인덱스 생성
        index_config = create_unified_index_config(target_index, dim)
        index_client.create_index(index_config)

        print(f"✅ Successfully created optimized RAPTOR index: {target_index}")
        return True

    except Exception as e:
        print(f"❌ Error creating index: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_documents_to_unified_index(
    search_client: SearchClient, documents: List[Dict[str, Any]], batch_size: int = 100
) -> bool:
    """통합 인덱스에 문서들 업로드"""
    try:
        print(f"📤 Uploading {len(documents)} optimized RAPTOR nodes...")

        total_batches = (len(documents) + batch_size - 1) // batch_size
        successful_uploads = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                result = search_client.upload_documents(documents=batch)

                success_count = sum(1 for r in result if r.succeeded)
                failed_count = len(batch) - success_count
                successful_uploads += success_count

                print(
                    f"   📦 Batch {batch_num}/{total_batches}: {success_count} success, {failed_count} failed"
                )

                if failed_count > 0:
                    for r in result:
                        if not r.succeeded:
                            print(
                                f"      ❌ Failed to upload document {r.key}: {r.error_message}"
                            )

            except Exception as e:
                print(f"   ❌ Error uploading batch {batch_num}: {e}")

        print(
            f"✅ Upload completed: {successful_uploads}/{len(documents)} documents uploaded successfully"
        )
        return successful_uploads == len(documents)

    except Exception as e:
        print(f"❌ Error in upload process: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_optimized_raptor(
    terms_index: str, raptor_index: str, target_index: str, dim: int = 3072
):
    """원본 약관 + RAPTOR 요약 노드만으로 최적화된 RAPTOR 트리 생성"""
    try:
        # Azure 클라이언트 생성
        index_client, endpoint, credential = create_azure_clients()

        # 통합 인덱스 생성
        if not create_unified_index(index_client, target_index, dim):
            return False

        target_search_client = SearchClient(
            endpoint=endpoint, index_name=target_index, credential=credential
        )

        all_raptor_nodes = []

        # 1. RAPTOR 요약 노드들만 가져오기 (리프 노드 제외)
        print(f"\n🌳 Processing RAPTOR summary nodes (layer > 0) from: {raptor_index}")
        raptor_search_client = SearchClient(
            endpoint=endpoint, index_name=raptor_index, credential=credential
        )

        raptor_documents = fetch_all_documents(raptor_search_client)
        max_node_index = get_max_node_index(raptor_documents)

        # 요약 노드만 필터링
        summary_nodes = []
        skipped_leaf_count = 0

        print(f"🔧 Filtering RAPTOR summary nodes (excluding leaf nodes)...")
        for doc in raptor_documents:
            normalized_doc = normalize_raptor_summary_node(doc)
            if normalized_doc is not None:
                summary_nodes.append(normalized_doc)
            else:
                skipped_leaf_count += 1

        print(f"   ✅ Kept {len(summary_nodes)} summary nodes")
        print(f"   🗑️  Skipped {skipped_leaf_count} unnecessary leaf nodes")

        all_raptor_nodes.extend(summary_nodes)

        # 2. 약관 데이터를 RAPTOR 리프 노드로 변환
        print(f"\n📄 Converting terms data to RAPTOR leaf nodes from: {terms_index}")
        terms_search_client = SearchClient(
            endpoint=endpoint, index_name=terms_index, credential=credential
        )

        terms_documents = fetch_all_documents(terms_search_client)

        print(
            f"🔧 Converting {len(terms_documents)} terms documents to RAPTOR leaf nodes..."
        )
        current_node_index = max_node_index + 1

        for doc in terms_documents:
            raptor_leaf = convert_terms_to_raptor_leaf(doc, current_node_index)
            all_raptor_nodes.append(raptor_leaf)
            current_node_index += 1

        # 3. 최적화된 인덱스에 업로드
        print(f"\n📤 Uploading {len(all_raptor_nodes)} optimized RAPTOR nodes...")
        success = upload_documents_to_unified_index(
            target_search_client, all_raptor_nodes
        )

        if success:
            print(f"\n🎉 Successfully created optimized RAPTOR tree in {target_index}")
            print(f"   📄 Terms converted to leaf nodes: {len(terms_documents)}")
            print(f"   🌳 RAPTOR summary nodes: {len(summary_nodes)}")
            print(f"   📊 Total optimized nodes: {len(all_raptor_nodes)}")
            print(f"   🗑️  Eliminated redundant leaf nodes: {skipped_leaf_count}")
            print(f"\n💡 최적화된 구조:")
            print(
                f"   └─ Layer 0 (Leaf): 원본 약관 텍스트만 ({len(terms_documents)}개)"
            )
            print(f"   └─ Layer 1+: RAPTOR 계층 요약만 ({len(summary_nodes)}개)")
        else:
            print(f"\n❌ Failed to create optimized RAPTOR tree")

        return success

    except Exception as e:
        print(f"❌ Error in optimization process: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create optimized RAPTOR tree (terms as leaf + summary nodes only)"
    )
    parser.add_argument("--terms-index", required=True, help="Source terms index name")
    parser.add_argument(
        "--raptor-index", required=True, help="Source RAPTOR index name"
    )
    parser.add_argument(
        "--target-index", required=True, help="Target optimized RAPTOR index name"
    )
    parser.add_argument(
        "--dim", type=int, default=3072, help="Vector dimension (default: 3072)"
    )

    args = parser.parse_args()

    print("🚀 Optimized RAPTOR Tree Builder")
    print("=" * 50)
    print(f"Terms Index: {args.terms_index}")
    print(f"RAPTOR Index: {args.raptor_index}")
    print(f"Target Index: {args.target_index}")
    print(f"Vector Dimension: {args.dim}")
    print("=" * 50)
    print("💡 최적화 전략:")
    print("   • 원본 약관 데이터 → 리프 노드 (Layer 0)")
    print("   • RAPTOR 요약 노드만 유지 (Layer 1+)")
    print("   • 중복 리프 노드 제거")
    print("=" * 50)

    success = create_optimized_raptor(
        args.terms_index, args.raptor_index, args.target_index, args.dim
    )

    if success:
        print("\n✅ Optimized RAPTOR tree creation completed successfully!")
        print("\n🔍 이제 다음과 같이 검색할 수 있습니다:")
        print("  • 원본 텍스트: layer eq 0")
        print("  • 요약 노드: layer gt 0")
        print("  • 루트 노드: is_root eq true")
        print("  • 특정 상품: source_title eq '상품명'")
        sys.exit(0)
    else:
        print("\n❌ Optimized RAPTOR tree creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
