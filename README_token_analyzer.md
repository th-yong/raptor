# Token & Cost Analyzer for RAPTOR Trees

pkl 파일에서 RAPTOR 트리구조를 불러와서 각 파일별 임베딩 및 요약 과정에서의 토큰량과 비용을 계산하는 도구입니다.

## 기능

- **임베딩 토큰량**: 각 노드의 텍스트를 임베딩할 때 사용되는 토큰 수
- **요약 입력 토큰량**: 요약 과정에서 입력으로 사용되는 토큰 수 (자식 노드들의 텍스트 + 시스템 프롬프트)
- **요약 출력 토큰량**: 요약 결과로 생성되는 토큰 수
- **비용 계산**: Azure OpenAI 가격 기준으로 계산된 비용 (USD)

## 사용법

### 1. 모든 pkl 파일 분석 (기본)

```bash
uv run token_cost_analyzer.py
```

- `results/` 디렉토리의 모든 pkl 파일을 분석
- 결과를 `token_cost_analysis.csv`에 저장
- 요약 통계 출력

### 2. 특정 디렉토리의 파일들 분석

```bash
uv run token_cost_analyzer.py --directory /path/to/pkl/files
```

### 3. 단일 파일 분석

```bash
uv run token_cost_analyzer.py --file results/현대해상3퇴직연금상품약관.pkl
```

### 4. 상세 정보와 함께 단일 파일 분석

```bash
uv run token_cost_analyzer.py --file results/현대해상3퇴직연금상품약관.pkl --detailed
```

### 5. 커스텀 출력 파일명 지정

```bash
uv run token_cost_analyzer.py --output my_analysis.csv
```

## 출력 정보

### CSV 파일 컬럼
- `filename`: 파일명
- `embedding_tokens`: 임베딩 토큰 수
- `summary_input_tokens`: 요약 입력 토큰 수
- `summary_output_tokens`: 요약 출력 토큰 수
- `total_tokens`: 총 토큰 수
- `embedding_cost_usd`: 임베딩 비용 (USD)
- `summary_cost_usd`: 요약 비용 (USD)
- `total_cost_usd`: 총 비용 (USD)
- `total_nodes`: 총 노드 수
- `num_layers`: 레이어 수
- `root_nodes`: 루트 노드 수
- `leaf_nodes`: 리프 노드 수

### 콘솔 출력
- 전체 통계 요약
- 파일별 비용 순위
- 평균 통계
- 최고/최저 비용 파일

## 가격 정보 (2024년 기준)

- **text-embedding-3-large**: $0.00013 per 1K tokens
- **gpt-4o**: 
  - Input: $0.0025 per 1K tokens
  - Output: $0.01 per 1K tokens

## 예시 출력

```
📊 TOKEN & COST ANALYSIS SUMMARY
================================================================================
📁 Total Files Analyzed: 40
🔢 Total Tokens: 2,450,123
   - Embedding Tokens: 1,200,456
   - Summary Input Tokens: 890,234
   - Summary Output Tokens: 359,433
💰 Total Cost: $0.012345 USD
   - Embedding Cost: $0.000156 USD
   - Summary Cost: $0.012189 USD

📈 Average per File:
   - Tokens: 61,253
   - Cost: $0.000309 USD
   - Nodes: 45.2
   - Layers: 3.8

🔝 Highest Cost File:
   - 현대해상3퇴직연금상품약관: $0.001234 USD (98,765 tokens)
🔻 Lowest Cost File:
   - KB손보1상품요약서: $0.000045 USD (3,456 tokens)
```

## 주의사항

1. tiktoken 라이브러리가 필요합니다: `uv add tiktoken`
2. pandas 라이브러리가 필요합니다: `uv add pandas`
3. pkl 파일이 올바른 RAPTOR Tree 객체여야 합니다
4. 가격 정보는 2024년 기준이며, 실제 Azure OpenAI 가격과 다를 수 있습니다