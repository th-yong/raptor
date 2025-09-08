#!/usr/bin/env python3
"""
기존 약관 인덱스 필드 + RAPTOR 필드를 통합한 스키마 정의
"""

from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
    ComplexField,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
)


def make_vector_field(name: str, dimensions: int, profile: str = "embedding_config"):
    """
    주어진 이름과 임베딩 차원, 그리고 벡터 검색 프로필을 기반으로
    `SearchField` 객체를 생성하여 반환합니다.

    Args:
        name (str): 필드 이름.
        dimensions (int): 임베딩 벡터의 차원 수.
        profile (str, optional): 벡터 검색 프로파일 이름. 기본값은 `"embedding_config"` 입니다.

    Returns:
        SearchField: 벡터 검색에 사용될 필드 객체.
    """
    return SearchField(
        name=name,
        type=SearchFieldDataType.Collection("Edm.Half"),
        vector_search_dimensions=dimensions,
        vector_search_profile_name=profile,
    )


def get_unified_fields(dim: int = 3072):
    """
    기존 약관 인덱스 필드들 + RAPTOR 필드들을 통합한 스키마를 반환합니다.

    기존 약관 필드:
    - id, doc_type, source_title, page_number
    - info (복합 필드: filename, create_date, update_date)
    - KO_content, KO_content_vector

    추가되는 RAPTOR 필드:
    - node_index, layer, is_root, is_leaf, children
    - created_at (RAPTOR 생성 시간)

    Args:
        dim (int): 임베딩 벡터 차원 수 (기본값: 3072)

    Returns:
        list[SearchField]: 통합 필드 목록.
    """
    return [
        # === 기존 약관 인덱스 필드들 ===
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="source_title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True,  # RAPTOR에서는 sortable=True로 사용
            facetable=True,  # 상품별 그룹핑을 위해 facetable 추가
            analyzer_name="ko.lucene",
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        ComplexField(
            name="info",
            fields=[
                SimpleField(
                    name="filename", type=SearchFieldDataType.String, filterable=True
                ),
                SimpleField(
                    name="create_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True,
                ),
                SimpleField(
                    name="update_date",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True,
                ),
            ],
        ),
        SearchField(
            name="KO_content",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            analyzer_name="ko.lucene",
        ),
        make_vector_field("KO_content_vector", dim),
        # === 추가되는 RAPTOR 전용 필드들 ===
        SimpleField(
            name="node_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="layer", type=SearchFieldDataType.Int32, filterable=True, sortable=True
        ),
        SimpleField(
            name="is_root",
            type=SearchFieldDataType.Boolean,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="is_leaf",
            type=SearchFieldDataType.Boolean,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="children",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="created_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
    ]


def create_unified_index_config(index_name: str, dim: int = 3072):
    """
    통합 인덱스 설정을 생성합니다.

    Args:
        index_name (str): 인덱스 이름
        dim (int): 임베딩 벡터 차원 수

    Returns:
        SearchIndex: 완전한 인덱스 설정
    """
    # 벡터 검색 설정
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="embedding_config", algorithm_configuration_name="hnsw_config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw_config",
                parameters={
                    "metric": "cosine",
                    "m": 16,
                    "efConstruction": 400,
                    "efSearch": 500,
                },
            )
        ],
    )

    return SearchIndex(
        name=index_name, fields=get_unified_fields(dim), vector_search=vector_search
    )


def get_field_mapping():
    """
    기존 약관 데이터와 RAPTOR 데이터를 통합할 때 사용할 필드 매핑 정보를 반환합니다.

    Returns:
        dict: 필드 매핑 정보
    """
    return {
        "common_fields": ["id", "source_title", "KO_content", "KO_content_vector"],
        "terms_only_fields": ["doc_type", "page_number", "info"],
        "raptor_only_fields": [
            "node_index",
            "layer",
            "is_root",
            "is_leaf",
            "children",
            "created_at",
        ],
        "field_defaults": {
            # 약관 데이터에 RAPTOR 필드가 없을 때 기본값
            "node_index": None,
            "layer": None,
            "is_root": None,
            "is_leaf": None,
            "children": None,
            "created_at": None,
            # RAPTOR 데이터에 약관 필드가 없을 때 기본값
            "doc_type": "raptor_node",
            "page_number": None,
            "info": None,
        },
    }


if __name__ == "__main__":
    # 사용 예시
    print("=== 통합 인덱스 스키마 ===")
    fields = get_unified_fields()

    print(f"총 {len(fields)}개 필드:")
    for field in fields:
        field_info = f"  - {field.name}: {field.type}"
        attributes = []

        if hasattr(field, "key") and field.key:
            attributes.append("키")
        if hasattr(field, "searchable") and field.searchable:
            attributes.append("검색가능")
        if hasattr(field, "filterable") and field.filterable:
            attributes.append("필터가능")
        if hasattr(field, "sortable") and field.sortable:
            attributes.append("정렬가능")
        if hasattr(field, "facetable") and field.facetable:
            attributes.append("패싯가능")

        if attributes:
            field_info += f" ({', '.join(attributes)})"
        print(field_info)

    print("\n=== 필드 매핑 정보 ===")
    mapping = get_field_mapping()
    for category, fields in mapping.items():
        if category != "field_defaults":
            print(f"{category}: {fields}")
