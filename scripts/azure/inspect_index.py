#!/usr/bin/env python3
"""
Azure AI Search 인덱스 통합 분석 도구
- 일반 인덱스 정보 확인
- RAPTOR 트리 구조 분석
- 상품별 분포 분석
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


def inspect_index(index_name, analyze_raptor=False):
    """특정 인덱스의 상세 정보 조회"""
    load_dotenv()

    endpoint = os.getenv("durable-azure-search-endpoint")
    api_key = os.getenv("durable-azure-search-api-key")

    if not endpoint or not api_key:
        print("❌ Azure Search endpoint and API key must be set in .env file")
        return

    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    search_client = SearchClient(
        endpoint=endpoint, index_name=index_name, credential=credential
    )

    try:
        print(f"🔍 인덱스 '{index_name}' 상세 정보")
        print("=" * 60)

        # 인덱스 스키마 확인
        index = index_client.get_index(index_name)
        print(f"📄 스키마 정보:")
        print("-" * 30)

        for field in index.fields:
            field_info = f"   - {field.name}: {field.type}"
            attributes = []
            if hasattr(field, 'key') and field.key:
                attributes.append("키")
            if hasattr(field, 'searchable') and field.searchable:
                attributes.append("검색가능")
            if hasattr(field, 'filterable') and field.filterable:
                attributes.append("필터가능")
            if hasattr(field, 'sortable') and field.sortable:
                attributes.append("정렬가능")
            if hasattr(field, 'facetable') and field.facetable:
                attributes.append("패싯가능")
            if attributes:
                field_info += f" ({', '.join(attributes)})"
            print(field_info)

        # 기본 문서 통계
        basic_analysis(search_client)
        
        # RAPTOR 트리 분석 (옵션)
        if analyze_raptor:
            raptor_analysis(search_client, index_name)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def basic_analysis(search_client):
    """기본 인덱스 분석"""
    # 문서 통계
    print(f"\n📊 문서 통계:")
    print("-" * 30)

    # 전체 문서 수 확인
    results = search_client.search(search_text="*", include_total_count=True, top=0)
    total_count = results.get_count()
    print(f"   전체 문서 수: {total_count}")

    # 샘플 문서 3개 확인
    print(f"\n📝 샘플 문서 (최대 5개):")
    print("-" * 30)

    sample_results = search_client.search(search_text="*", top=5)
    sample_docs = list(sample_results)

    for i, doc in enumerate(sample_docs, 1):
        print(f"\n   문서 {i}:")
        for key, value in doc.items():
            if key.startswith("@"):
                continue  # Azure Search 메타데이터 필드 제외

            if isinstance(value, str):
                if len(value) > 150:
                    value = value[:150] + "..."
            elif isinstance(value, list):
                if len(value) > 10:
                    value = f"[리스트, {len(value)}개 항목]"
                elif len(value) > 0 and isinstance(value[0], (int, float)):
                    value = f"[벡터, {len(value)}개 차원]"

            print(f"     {key}: {value}")


def raptor_analysis(search_client, index_name):
    """RAPTOR 트리 구조 분석"""
    print(f"\n🌳 RAPTOR 트리 분석")
    print("=" * 50)
    
    # RAPTOR 인덱스인지 확인
    sample_doc = list(search_client.search(search_text="*", top=1))
    if not sample_doc or 'layer' not in sample_doc[0]:
        print("❌ 이 인덱스는 RAPTOR 구조가 아닙니다.")
        return
    
    total_results = search_client.search(
        search_text='*', 
        top=0,
        include_total_count=True
    )
    total_count = total_results.get_count()
    print(f"� 총 노드 수: {total_count}")
    
    # 계층별 노드 개수 확인
    print(f"\n📊 계층별 노드 분포:")
    print("-" * 30)
    
    layer_counts = {}
    for layer in range(0, 5):  # 0~4층까지 확인
        results = search_client.search(
            search_text='*', 
            filter=f"layer eq {layer}", 
            top=0,
            include_total_count=True
        )
        count = results.get_count()
        if count > 0:
            layer_counts[layer] = count
            print(f"  Layer {layer}: {count}개 노드")
    
    # 리프 노드 분석
    if 0 in layer_counts:
        print(f"\n🍃 리프 노드 (Layer 0) 분석:")
        print("-" * 30)
        
        # 원본 약관 데이터 개수 (page_number가 있는 것들)
        try:
            terms_with_page = search_client.search(
                search_text='*', 
                filter="layer eq 0 and page_number ne null", 
                top=0,
                include_total_count=True
            )
            terms_count = terms_with_page.get_count()
            print(f"  📄 원본 약관 데이터: {terms_count}개 (페이지 정보 있음)")
        except:
            print(f"  📄 원본 약관 데이터: 확인 불가")
        
        print(f"  📊 총 리프 노드: {layer_counts.get(0, 0)}개")
        
        # 리프 노드 샘플
        leaf_results = search_client.search(
            search_text='*', 
            filter="layer eq 0", 
            top=2
        )
        
        for i, doc in enumerate(leaf_results, 1):
            print(f"\n  리프 노드 {i}:")
            print(f"    - 상품: {doc.get('source_title', 'N/A')}")
            print(f"    - 페이지: {doc.get('page_number', 'N/A')}")
            print(f"    - 노드 인덱스: {doc.get('node_index', 'N/A')}")
            
            # 내용 일부 표시
            content = doc.get('KO_content', '')
            if content:
                content_preview = content[:100] + '...' if len(content) > 100 else content
                print(f"    - 내용: {content_preview}")
    
    # 요약 노드 분석
    summary_layers = [layer for layer in layer_counts.keys() if layer > 0]
    if summary_layers:
        print(f"\n🌿 요약 노드 분석:")
        print("-" * 30)
        
        for layer in sorted(summary_layers):
            results = search_client.search(
                search_text='*', 
                filter=f"layer eq {layer}", 
                top=1
            )
            
            sample_doc = list(results)
            if sample_doc:
                doc = sample_doc[0]
                print(f"  Layer {layer} 샘플:")
                print(f"    - 상품: {doc.get('source_title', 'N/A')}")
                print(f"    - 노드 인덱스: {doc.get('node_index', 'N/A')}")
                print(f"    - 루트: {doc.get('is_root', False)}")
                
                # 자식 노드 정보
                children_str = doc.get('children', '[]')
                if children_str and children_str != '[]':
                    try:
                        children_list = eval(children_str)
                        print(f"    - 자식 수: {len(children_list)}개")
                    except:
                        print(f"    - 자식 정보: 파싱 오류")
    
    # 루트 노드 확인
    try:
        root_results = search_client.search(
            search_text='*', 
            filter="is_root eq true", 
            top=0,
            include_total_count=True
        )
        root_count = root_results.get_count()
        print(f"\n🌳 루트 노드: {root_count}개")
    except:
        print(f"\n🌳 루트 노드: 확인 불가")
    
    # 상품별 분포
    print(f"\n📋 상품별 노드 분포:")
    print("-" * 30)
    
    try:
        # source_title 필드의 고유 값들 확인
        facet_results = search_client.search(
            search_text="*", 
            facets=["source_title"], 
            top=0
        )
        
        facets = facet_results.get_facets()
        if "source_title" in facets:
            products = [f["value"] for f in facets["source_title"][:10]]
            
            for product in products:
                try:
                    total_results = search_client.search(
                        search_text='*', 
                        filter=f"source_title eq '{product}'", 
                        top=0,
                        include_total_count=True
                    )
                    total = total_results.get_count()
                    
                    # 리프 노드 수
                    leaf_results = search_client.search(
                        search_text='*', 
                        filter=f"source_title eq '{product}' and layer eq 0", 
                        top=0,
                        include_total_count=True
                    )
                    leaf_count = leaf_results.get_count()
                    
                    # 요약 노드 수
                    summary_count = total - leaf_count
                    
                    print(f"  {product[:35]}{'...' if len(product) > 35 else ''}:")
                    print(f"    - 총 노드: {total}개")
                    print(f"    - 리프: {leaf_count}개")
                    print(f"    - 요약: {summary_count}개")
                    print()
                except Exception as e:
                    print(f"  {product[:35]}: 분석 실패 ({e})")
        else:
            print("  상품별 분포 정보를 가져올 수 없습니다.")
    except Exception as e:
        print(f"  상품별 분포 분석 실패: {e}")


def list_all_indexes():
    """모든 인덱스 목록 표시"""
    load_dotenv()
    
    endpoint = os.getenv("durable-azure-search-endpoint")
    api_key = os.getenv("durable-azure-search-api-key")
    
    if not endpoint or not api_key:
        print("❌ Azure Search endpoint and API key must be set in .env file")
        return
    
    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    
    try:
        print(f"🔍 Azure AI Search 엔드포인트: {endpoint}")
        print("📋 인덱스 목록:")
        print("-" * 50)
        
        indexes = index_client.list_indexes()
        index_list = list(indexes)
        
        for i, index in enumerate(index_list, 1):
            print(f"   {i}. {index.name}")
        
        print(f"\n✅ 총 {len(index_list)}개의 인덱스가 있습니다.")
        
    except Exception as e:
        print(f"❌ 인덱스 목록 조회 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="Azure AI Search 인덱스 통합 분석 도구")
    parser.add_argument("--index", help="분석할 인덱스 이름")
    parser.add_argument("--list", action="store_true", help="모든 인덱스 목록 표시")
    parser.add_argument("--raptor", action="store_true", help="RAPTOR 트리 구조 분석 포함")
    
    args = parser.parse_args()
    
    if args.list:
        list_all_indexes()
    elif args.index:
        inspect_index(args.index, args.raptor)
    else:
        # 호환성을 위해 기존 방식도 지원
        if len(sys.argv) == 2:
            index_name = sys.argv[1]
            inspect_index(index_name, False)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
