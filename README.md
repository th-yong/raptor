<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="raptor.jpg">
</picture>

# RAPTOR: 보험 약관 특화 계층적 검색 시스템

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval)를 보험 약관 문서에 특화하여 최적화한 시스템입니다. 페이지 인식 클러스터링과 o3 모델 최적화를 통해 약관 검색 성능을 대폭 향상시켰습니다.

[![Paper](https://img.shields.io/badge/Paper-RAPTOR-blue)](https://arxiv.org/abs/2401.18059)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![Azure](https://img.shields.io/badge/Azure-OpenAI-orange)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

## 🚀 주요 최적화 사항

### 🧠 o3 모델 토큰 최적화
- **개별 청크**: 2,000 tokens (기존 100에서 20배 증가)
- **요약 길이**: 4,000 tokens (기존 800에서 5배 증가)  
- **클러스터 크기**: 25,000 tokens (기존 3,500에서 7배 증가)
- **정보 손실 최소화**로 검색 정확도 대폭 향상

### 📄 페이지 인식 클러스터링 (옵션)
- **기본 모드**: 페이지 제약 없이 자유로운 클러스터링 (기본값)
- **페이지 제약 모드**: 논리적으로 연관된 페이지만 클러스터링
- **max_page_gap**: 페이지 제약 모드에서 최대 10페이지 간격 제한
- **약관 특성**: 서로 다른 조항들의 무분별한 클러스터링 방지 (선택사항)

### 🎯 약관 특화 요약 프롬프트
- **검색 최적화**: 키워드, 개념, 조건을 명확히 포함
- **구조 보존**: 조항 번호, 단락 구조 유지
- **관계 명시**: 조건문, 예외사항, 참조 관계 강조

### 🌳 최적화된 트리 구조
- **2-level structure**: 과도한 일반화 방지
- **num_layers=1**: 원본 정보 보존과 요약의 균형
- **Azure AI Search**: 클라우드 검색

## 📦 설치

```bash
# Python 3.11+ 필요
git clone https://github.com/th-yong/raptor.git
cd raptor
uv sync  # pyproject.toml과 uv.lock 사용
```

## ⚙️ 환경 설정

1. `.env` 파일 생성:
```bash
# Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01

# 모델 배포 이름
AZURE_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_COMPLETION_DEPLOYMENT_NAME=o3-mini

# Azure AI Search (선택사항)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
```

2. 환경 검증:
```bash
uv run check_o3_optimization.py
```

## � 사용법

### 1. 환경 검증 (필수 첫 단계)

```bash
# 환경 및 설정 검증 (가장 중요!)
uv run check_o3_optimization.py
```

### 2. 편의 스크립트 (간단한 실행)

```bash
# 문서 처리
uv run process_documents.py

# 예시 실행  
uv run run_example.py input/insurance_terms.csv

# Azure 인덱스 생성
uv run create_index.py

# 토큰 분석
uv run analyze_tokens.py
```

### 3. 세부 제어 (직접 스크립트 실행)

```bash
# 페이지 제약 처리 (약관 문서 최적화)  
uv run scripts/examples/example_page_aware_raptor.py input/insurance_terms.csv --max-page-gap 10

# 문서 처리 (고급 옵션)
uv run scripts/processing/process_and_merge_documents.py --force

# Azure 관련
uv run scripts/azure/create_optimized_raptor.py
uv run scripts/azure/inspect_index.py

# 분석 도구
uv run scripts/analysis/token_cost_analyzer.py --detailed
```

### 4. 프로그래밍 인터페이스

```python
from config.page_aware_config import create_page_aware_config
from raptor import RetrievalAugmentation
import pandas as pd

# 기본 설정 (페이지 제약 없음)
config = create_page_aware_config(use_page_constraint=False)

# 또는 페이지 제약 모드 (약관 문서에 최적화)
config = create_page_aware_config(max_page_gap=10, use_page_constraint=True)

# CSV 데이터 (page_number 컬럼은 페이지 제약 모드에서만 필요)
df = pd.read_csv("insurance_terms.csv")

# 청크 준비
chunks = []
for _, row in df.iterrows():
    chunk_data = {"text": row["content"]}
    # 페이지 제약 모드 사용 시에만 page_number 추가
    if "page_number" in df.columns:
        chunk_data["page_number"] = row["page_number"]
    chunks.append(chunk_data)

# RAPTOR 처리
ra = RetrievalAugmentation(config=config)
ra.add_documents(chunks)

# 검색
result = ra.answer_question("보장 한도는 얼마인가요?")
```

## 💡 권장 워크플로

1. **첫 실행**: `uv run check_o3_optimization.py` (환경 검증)
2. **문서 처리**: `uv run process_documents.py` (편의 스크립트)
3. **예시 테스트**: `uv run run_example.py input/data.csv`
4. **고급 기능**: `scripts/` 폴더의 개별 스크립트 사용

## 📊 프로젝트 구조

### 🏗️ 정리된 폴더 구조

```
raptor/
├── � check_o3_optimization.py      # ⭐ 메인 환경 검증 스크립트
├── �📁 config/                       # 설정 파일들
│   ├── page_aware_config.py         # 페이지 인식 RAPTOR 설정
│   └── unified_index_schema.py      # Azure Search 스키마
├── 📁 scripts/                      # 기능별 스크립트들
│   ├── 📁 processing/               # 문서 처리
│   │   └── process_and_merge_documents.py
│   ├── 📁 azure/                    # Azure 관련  
│   │   ├── create_optimized_raptor.py
│   │   ├── process_azure_search.py
│   │   └── inspect_index.py
│   ├── 📁 analysis/                 # 분석 도구
│   │   ├── token_cost_analyzer.py
│   │   └── resummary_with_different_models.py
│   └── 📁 examples/                 # 사용 예시
│       └── example_page_aware_raptor.py
├── � 편의 실행 스크립트들            # 간편한 실행을 위한 래퍼들
│   ├── process_documents.py         # → scripts/processing/
│   ├── run_example.py              # → scripts/examples/
│   ├── create_index.py             # → scripts/azure/
│   └── analyze_tokens.py           # → scripts/analysis/
└── 📁 기타 폴더들
    ├── raptor/                     # 핵심 라이브러리
    ├── utils/                      # 유틸리티
    ├── input/                      # 입력 데이터
    └── results/                    # 결과 파일들
```

## 📊 스크립트 구성

### 핵심 스크립트
- **`check_o3_optimization.py`**: 환경 검증 및 최적화 설정 확인 (메인)
- **`config/page_aware_config.py`**: 페이지 인식 RAPTOR 설정 팩토리  
- **`scripts/examples/example_page_aware_raptor.py`**: 페이지 인식 처리 예시
- **`scripts/processing/process_and_merge_documents.py`**: 대량 문서 통합 처리

### 유틸리티 스크립트
- **`scripts/azure/create_optimized_raptor.py`**: Azure Search 인덱스 생성
- **`config/unified_index_schema.py`**: 통합 인덱스 스키마 정의
- **`scripts/analysis/token_cost_analyzer.py`**: 토큰 비용 분석
- **`scripts/analysis/resummary_with_different_models.py`**: 다양한 모델로 재요약

### 편의 스크립트 (루트 레벨)
- **`process_documents.py`**: 문서 처리 편의 실행
- **`run_example.py`**: 예시 실행 편의 스크립트
- **`create_index.py`**: 인덱스 생성 편의 실행
- **`analyze_tokens.py`**: 토큰 분석 편의 실행

## 🔍 핵심 클래스

### PageAwareRAPTORClustering (옵션)
```python
from raptor.cluster_utils import PageAwareRAPTORClustering, RAPTOR_Clustering

# 기본 클러스터링 (페이지 제약 없음)
clustering = RAPTOR_Clustering(
    embedding_model=embedding_model,
    max_length_in_cluster=25000  # 토큰 제한
)

# 페이지 제약 클러스터링 (약관 문서 특화)
clustering = PageAwareRAPTORClustering(
    max_page_gap=10,  # 최대 페이지 간격
    embedding_model=embedding_model,
    max_length_in_cluster=25000  # 토큰 제한
)
```

### AzureSummarizationModel (최적화)
```python
from raptor.SummarizationModels import AzureSummarizationModel

model = AzureSummarizationModel(
    client_manager=azure_client,
    max_tokens=4000  # 5배 증가
)
```

## 📈 성능 향상

| 항목 | 기존 | 최적화 | 향상률 |
|------|------|--------|--------|
| 청크 토큰 | 100 | 2,000 | 20x |
| 요약 토큰 | 800 | 4,000 | 5x |
| 클러스터 토큰 | 3,500 | 25,000 | 7x |
| 트리 레벨 | 3+ | 1 | 단순화 |
| 페이지 제약 | 없음 | 옵션 | 유연성 |

## 🧪 검증 도구

```bash
# 설정 검증
uv run check_o3_optimization.py

# 토큰 비용 분석
uv run scripts/analysis/token_cost_analyzer.py
# 또는
uv run analyze_tokens.py

# Azure Search 인덱스 검사
uv run scripts/azure/inspect_index.py
```

## 📝 입력 데이터 형식

CSV 파일에 다음 컬럼이 필요합니다:

```csv
source_title,page_number,content
"무배당 심뇌혈관종합건강보험",1,"제1조 목적..."
"무배당 심뇌혈관종합건강보험",2,"제2조 정의..."
```

## 🎯 약관 특화 기능

1. **조항 구조 보존**: 조항 번호와 계층 구조 유지
2. **조건문 명시**: "만약", "단서", "예외" 등의 조건 관계 강조
3. **참조 관계**: 다른 조항 참조 시 명확한 연결 관계 표시
4. **키워드 추출**: 보험 용어, 보장 내용, 제외 사항 등 핵심 키워드 포함
5. **논리적 그룹핑**: 페이지 간격 제한으로 관련 조항만 클러스터링

## 🔗 참고 자료

- [원본 RAPTOR 논문](https://arxiv.org/abs/2401.18059)
- [Azure OpenAI Service](https://azure.microsoft.com/products/ai-services/openai-service)
- [UV 패키지 매니저](https://github.com/astral-sh/uv)

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE.txt](LICENSE.txt)를 참조하세요.
