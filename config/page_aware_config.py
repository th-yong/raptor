#!/usr/bin/env python3
"""
페이지 인식 RAPTOR 클러스터링 설정

약관 문서에 최적화된 설정:
- 페이지 간격이 큰 내용들은 클러스터링하지 않음
- 연속된 페이지 범위 내에서만 클러스터링 수행
"""

from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from raptor.cluster_utils import PageAwareRAPTORClustering, RAPTOR_Clustering
from raptor.SummarizationModels import AzureSummarizationModel
from raptor.EmbeddingModels import AzureEmbeddingModel
from utils.llm_manager import AzureAIClientManager


def create_page_aware_config(max_page_gap=None, use_page_constraint=False):
    """
    RAPTOR 설정 생성 (약관 문서 최적화, 페이지 제약 옵션)

    Args:
        max_page_gap (int, optional): 클러스터링을 허용할 최대 페이지 간격 (기본: None)
        use_page_constraint (bool): 페이지 제약 사용 여부 (기본: False)

    특징:
        - 레벨 1까지만 생성 (과도한 일반화 방지)
        - 페이지 기반 클러스터링은 옵션 (use_page_constraint=True 시 활성화)
        - 약관 특화 요약 프롬프트 사용
    """

    # Azure 클라이언트 설정
    client = AzureAIClientManager()

    # 임베딩 모델 설정 (약관에 특화된 모델)
    embedding_models = {"azure_embedding": AzureEmbeddingModel(client=client)}

    # 요약 모델 설정 (개선된 약관 요약 프롬프트 사용)
    summarization_model = AzureSummarizationModel(client=client)

    # 클러스터링 알고리즘 및 매개변수 설정
    if use_page_constraint and max_page_gap is not None:
        # 페이지 제약 클러스터링 사용
        clustering_algorithm = PageAwareRAPTORClustering
        clustering_params = {
            "max_page_gap": max_page_gap,  # 최대 페이지 간격
            "max_length_in_cluster": 25000,  # o3 최적화: 클러스터 내 최대 길이 대폭 확장
            "reduction_dimension": 7,
            "threshold": 0.1,
            "verbose": True,
        }
    else:
        # 기본 RAPTOR 클러스터링 사용 (페이지 제약 없음)
        clustering_algorithm = RAPTOR_Clustering
        clustering_params = {
            "max_length_in_cluster": 25000,  # o3 최적화: 클러스터 내 최대 길이 대폭 확장
            "reduction_dimension": 7,
            "threshold": 0.1,
            "verbose": True,
        }

    # RAPTOR 설정
    config = ClusterTreeConfig(
        # 트리 구조 설정 (약관에 최적화: 레벨1까지만)
        num_layers=2,  # 레벨 0(리프) + 레벨 1만 생성
        threshold=0.5,
        top_k=3,
        selection_mode="top_k",
        summarization_length=4000,  # o3 최적화: 매우 상세한 요약
        max_tokens=2000,  # o3 최적화: 개별 청크 크기 확장
        # 클러스터링 설정
        clustering_algorithm=clustering_algorithm,
        cluster_embedding_model="azure_embedding",  # 클러스터링에 사용할 임베딩 모델 지정
        clustering_params=clustering_params,
        # 모델 설정
        embedding_models=embedding_models,
        summarization_model=summarization_model,
    )

    return config


def create_page_aware_builder(max_page_gap=10, use_page_constraint=True):
    """페이지 인식 RAPTOR 빌더 생성 (하위 호환성을 위해 기본값 유지)"""
    config = create_page_aware_config(max_page_gap, use_page_constraint)
    return ClusterTreeBuilder(config)


def create_optimized_config(use_page_constraint=False, max_page_gap=10):
    """
    최적화된 RAPTOR 설정 생성 (새로운 인터페이스)

    Args:
        use_page_constraint (bool): 페이지 제약 사용 여부 (기본: False)
        max_page_gap (int): 페이지 제약 사용 시 최대 간격 (기본: 10)
    """
    return create_page_aware_config(max_page_gap, use_page_constraint)


# 사용 예시
if __name__ == "__main__":
    print("페이지 인식 RAPTOR 설정 (약관 최적화):")
    print("- 최대 페이지 간격: 10페이지")
    print("- 레벨 1까지만 생성 (과도한 일반화 방지)")
    print("- 연속된 페이지만 클러스터링")
    print("- 약관 최적화 요약 프롬프트")

    # 설정 생성
    config = create_page_aware_config(max_page_gap=10)
    print(f"\n설정 완료: {config.log_config()}")

    # 빌더 생성
    builder = create_page_aware_builder(max_page_gap=10)
    print("빌더 생성 완료")
