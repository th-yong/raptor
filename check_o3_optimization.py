#!/usr/bin/env python3
"""
o3 최적화 RAPTOR 설정 확인 스크립트

토큰 제한이 대폭 확장된 설정을 확인하고 환경 변수도 검증합니다.
"""

import os
from dotenv import load_dotenv
from config.page_aware_config import create_page_aware_config
from raptor.SummarizationModels import AzureSummarizationModel
from raptor.cluster_utils import PageAwareRAPTORClustering, RAPTOR_Clustering


def check_environment():
    """환경 변수 확인"""
    print("🔍 환경 변수 확인")
    print("=" * 30)

    load_dotenv()

    # 필수 환경 변수들
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_COMPLETION_DEPLOYMENT_NAME",
        "AZURE_EMBEDDING_DEPLOYMENT_NAME",
    ]

    optional_vars = ["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY"]

    print("📋 필수 환경 변수:")
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # API 키는 일부만 표시
            if "KEY" in var or "key" in var:
                display_value = (
                    f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "****"
                )
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: 설정되지 않음")
            all_good = False

    print("\n📋 선택적 환경 변수 (Azure AI Search):")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            if "key" in var:
                display_value = (
                    f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "****"
                )
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ⚠️ {var}: 설정되지 않음 (Azure Search 사용 시 필요)")

    return all_good


def check_optimized_settings():
    """최적화된 설정 확인"""
    print("🚀 o3 최적화 RAPTOR 설정 확인")
    print("=" * 60)

    # 1. 기본 설정 확인 (페이지 제약 없음)
    try:
        print("\n🔧 기본 설정 (페이지 제약 없음):")
        config_basic = create_page_aware_config(use_page_constraint=False)
        print("✅ 기본 Config 생성 성공")

        print(f"\n📊 트리 구조 설정:")
        print(f"  - num_layers: {config_basic.num_layers}")
        print(f"  - summarization_length: {config_basic.summarization_length}")
        print(f"  - max_tokens: {getattr(config_basic, 'max_tokens', 'N/A')}")

        print(f"\n🔧 클러스터링 설정:")
        print(f"  - Algorithm: {config_basic.clustering_algorithm.__name__}")
        print(f"  - Parameters: {config_basic.clustering_params}")

    except Exception as e:
        print(f"❌ 기본 Config 생성 실패: {e}")
        print("💡 환경 변수가 올바르게 설정되었는지 확인하세요.")
        return

    # 2. 페이지 제약 설정 확인
    try:
        print("\n🔧 페이지 제약 설정 (max_page_gap=10):")
        config_page = create_page_aware_config(
            max_page_gap=10, use_page_constraint=True
        )
        print("✅ 페이지 제약 Config 생성 성공")

        print(f"\n🔧 클러스터링 설정:")
        print(f"  - Algorithm: {config_page.clustering_algorithm.__name__}")
        print(f"  - Parameters: {config_page.clustering_params}")

        print(f"\n🤖 모델 설정:")
        print(
            f"  - Summarization Model: {type(config_page.summarization_model).__name__}"
        )
        print(f"  - Embedding Models: {list(config_page.embedding_models.keys())}")

    except Exception as e:
        print(f"❌ 페이지 제약 Config 생성 실패: {e}")
        return

    # 요약 모델 토큰 제한 확인
    print(f"\n📝 요약 모델 토큰 제한:")

    # Azure 모델 확인 (환경변수 없어도 클래스 정보는 확인 가능)
    import inspect

    try:
        azure_sig = inspect.signature(AzureSummarizationModel.summarize)
        azure_params = azure_sig.parameters
        azure_max_tokens = azure_params.get("max_tokens")
        if azure_max_tokens and azure_max_tokens.default:
            print(f"  - Azure 모델: {azure_max_tokens.default} tokens")
    except Exception as e:
        print(f"  - Azure 모델: 확인 실패 ({e})")

    # 클러스터링 알고리즘 토큰 제한 확인
    print(f"\n🔗 클러스터링 토큰 제한:")

    try:
        raptor_sig = inspect.signature(RAPTOR_Clustering.perform_clustering)
        raptor_params = raptor_sig.parameters
        raptor_max_length = raptor_params.get("max_length_in_cluster")
        if raptor_max_length and raptor_max_length.default:
            print(f"  - RAPTOR 클러스터: {raptor_max_length.default} tokens")
    except Exception as e:
        print(f"  - RAPTOR 클러스터: 확인 실패 ({e})")

    try:
        page_aware_sig = inspect.signature(PageAwareRAPTORClustering.perform_clustering)
        page_aware_params = page_aware_sig.parameters
        page_aware_max_length = page_aware_params.get("max_length_in_cluster")
        if page_aware_max_length and page_aware_max_length.default:
            print(f"  - PageAware 클러스터: {page_aware_max_length.default} tokens")
    except Exception as e:
        print(f"  - PageAware 클러스터: 확인 실패 ({e})")

    print(f"\n🎯 최적화 요약:")
    print(f"  - 개별 청크: 2,000 tokens (기존 100에서 20배 증가)")
    print(f"  - 요약 길이: 4,000 tokens (기존 800에서 5배 증가)")
    print(f"  - 클러스터 크기: 25,000 tokens (기존 3,500에서 7배 증가)")
    print(f"  - 레벨: 1단계만 (과도한 일반화 방지)")

    print(f"\n📄 페이지 제약 옵션:")
    print(f"  🔧 기본 모드: 페이지 제약 없음 (use_page_constraint=False)")
    print(f"     → 모든 청크가 자유롭게 클러스터링됨")
    print(f"  🔧 페이지 제약 모드: 최대 {10}페이지 간격 (use_page_constraint=True)")
    print(f"     → 논리적으로 연관된 페이지만 클러스터링됨")

    print(f"\n💡 사용법:")
    print(f"  # 기본 모드 (페이지 제약 없음)")
    print(f"  config = create_page_aware_config(use_page_constraint=False)")
    print(f"  ")
    print(f"  # 페이지 제약 모드")
    print(
        f"  config = create_page_aware_config(max_page_gap=10, use_page_constraint=True)"
    )

    print(f"\n💡 예상 효과:")
    print(f"  ✅ 정보 손실 최소화")
    print(f"  ✅ 더 상세하고 포괄적인 요약")
    print(f"  ✅ 검색 성능 대폭 향상")
    print(f"  ✅ 약관 문서 특성에 최적화 (페이지 제약 모드)")


def main():
    """메인 실행 함수"""
    print("🔧 RAPTOR 설정 및 환경 검증 도구")
    print("=" * 60)

    # 1. 환경 변수 확인
    env_ok = check_environment()

    # 2. 최적화 설정 확인
    check_optimized_settings()

    # 3. 종합 평가
    print(f"\n📋 종합 평가:")
    if env_ok:
        print("✅ 환경 설정 완료 - 모든 기능 사용 가능")
        print("🚀 페이지 인식 RAPTOR 처리 준비 완료!")
    else:
        print("⚠️ 환경 설정 미완료 - .env 파일을 확인하세요")
        print("📖 README.md의 환경 설정 섹션을 참조하세요")


if __name__ == "__main__":
    main()
