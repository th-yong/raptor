import pickle
import os
import tiktoken
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from raptor.tree_structures import Tree, Node

@dataclass
class TokenCostInfo:
    """토큰 사용량 및 비용 정보를 저장하는 클래스"""
    embedding_tokens: int = 0
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    total_tokens: int = 0
    embedding_cost: float = 0.0
    summary_cost: float = 0.0
    total_cost: float = 0.0

class TokenCostAnalyzer:
    """pkl 파일에서 트리를 로드하고 토큰량 및 비용을 분석하는 클래스"""
    
    # Azure OpenAI 가격 (USD per 1K tokens)
    PRICING = {
        "text-embedding-3-large": 0.00013,  # per 1K tokens
        "gpt-4o": {
            "input": 0.0025,   # per 1K input tokens
            "output": 0.01     # per 1K output tokens
        }
    }
    
    def __init__(self):
        # GPT-4o 토크나이저 사용
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산"""
        if not text or not isinstance(text, str):
            return 0
        return len(self.tokenizer.encode(text))
    
    def load_tree_from_pkl(self, pkl_path: str) -> Tree:
        """pkl 파일에서 트리를 로드"""
        try:
            with open(pkl_path, 'rb') as f:
                tree = pickle.load(f)
            if not isinstance(tree, Tree):
                raise ValueError(f"Loaded object is not a Tree instance: {type(tree)}")
            return tree
        except Exception as e:
            raise ValueError(f"Failed to load tree from {pkl_path}: {e}")
    
    def analyze_tree_tokens(self, tree: Tree) -> TokenCostInfo:
        """트리의 모든 노드를 분석하여 토큰량과 비용을 계산"""
        cost_info = TokenCostInfo()
        
        # 모든 노드 분석
        for node_id, node in tree.all_nodes.items():
            if not isinstance(node, Node):
                continue
                
            # 임베딩 토큰 계산 (각 노드의 텍스트)
            embedding_tokens = self.count_tokens(node.text)
            cost_info.embedding_tokens += embedding_tokens
            
            # 요약 토큰 계산 (leaf 노드가 아닌 경우, 즉 요약된 노드인 경우)
            if node.children:  # 자식이 있다면 요약된 노드
                # 자식 노드들의 텍스트를 합쳐서 입력 토큰 계산
                child_texts = []
                for child_id in node.children:
                    if child_id in tree.all_nodes:
                        child_node = tree.all_nodes[child_id]
                        child_texts.append(child_node.text)
                
                if child_texts:
                    # 입력: 자식 노드들의 텍스트 + 시스템 프롬프트
                    input_text = "\n".join(child_texts)
                    system_prompt = "당신은 보험약관에 대해 안전한 요약을 제공하는 유용한 어시스턴트입니다..."
                    user_prompt = f"다음 텍스트를 간결하게 요약해 주세요:\n\n{input_text}"
                    
                    input_tokens = (
                        self.count_tokens(system_prompt) + 
                        self.count_tokens(user_prompt)
                    )
                    
                    # 출력: 현재 노드의 텍스트 (요약 결과)
                    output_tokens = self.count_tokens(node.text)
                    
                    cost_info.summary_input_tokens += input_tokens
                    cost_info.summary_output_tokens += output_tokens
        
        # 총 토큰 계산
        cost_info.total_tokens = (
            cost_info.embedding_tokens + 
            cost_info.summary_input_tokens + 
            cost_info.summary_output_tokens
        )
        
        # 비용 계산
        cost_info.embedding_cost = (cost_info.embedding_tokens / 1000) * self.PRICING["text-embedding-3-large"]
        cost_info.summary_cost = (
            (cost_info.summary_input_tokens / 1000) * self.PRICING["gpt-4o"]["input"] +
            (cost_info.summary_output_tokens / 1000) * self.PRICING["gpt-4o"]["output"]
        )
        cost_info.total_cost = cost_info.embedding_cost + cost_info.summary_cost
        
        return cost_info
    
    def analyze_pkl_file(self, pkl_path: str) -> Tuple[str, TokenCostInfo, Dict[str, Any]]:
        """단일 pkl 파일을 분석"""
        filename = Path(pkl_path).stem
        
        try:
            tree = self.load_tree_from_pkl(pkl_path)
            cost_info = self.analyze_tree_tokens(tree)
            
            # 트리 구조 정보
            tree_info = {
                "total_nodes": len(tree.all_nodes),
                "num_layers": tree.num_layers,
                "root_nodes": len(tree.root_nodes),
                "leaf_nodes": len(tree.leaf_nodes),
                "layer_distribution": {layer: len(nodes) for layer, nodes in tree.layer_to_nodes.items()}
            }
            
            return filename, cost_info, tree_info
            
        except Exception as e:
            print(f"Error analyzing {pkl_path}: {e}")
            return filename, TokenCostInfo(), {}
    
    def analyze_all_pkl_files(self, pkl_directory: str = "results") -> pd.DataFrame:
        """디렉토리의 모든 pkl 파일을 분석하여 DataFrame으로 반환"""
        pkl_dir = Path(pkl_directory)
        pkl_files = list(pkl_dir.glob("*.pkl"))
        
        if not pkl_files:
            print(f"No pkl files found in {pkl_directory}")
            return pd.DataFrame()
        
        results = []
        
        print(f"Analyzing {len(pkl_files)} pkl files...")
        
        for i, pkl_file in enumerate(pkl_files, 1):
            print(f"[{i}/{len(pkl_files)}] Analyzing: {pkl_file.name}")
            
            filename, cost_info, tree_info = self.analyze_pkl_file(str(pkl_file))
            
            result = {
                "filename": filename,
                "embedding_tokens": cost_info.embedding_tokens,
                "summary_input_tokens": cost_info.summary_input_tokens,
                "summary_output_tokens": cost_info.summary_output_tokens,
                "total_tokens": cost_info.total_tokens,
                "embedding_cost_usd": round(cost_info.embedding_cost, 6),
                "summary_cost_usd": round(cost_info.summary_cost, 6),
                "total_cost_usd": round(cost_info.total_cost, 6),
                "total_nodes": tree_info.get("total_nodes", 0),
                "num_layers": tree_info.get("num_layers", 0),
                "root_nodes": tree_info.get("root_nodes", 0),
                "leaf_nodes": tree_info.get("leaf_nodes", 0),
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """분석 결과의 요약 통계를 출력"""
        if df.empty:
            print("No data to summarize")
            return
        
        print("\n" + "="*80)
        print("📊 TOKEN & COST ANALYSIS SUMMARY")
        print("="*80)
        
        # 전체 통계
        total_files = len(df)
        total_embedding_tokens = df['embedding_tokens'].sum()
        total_summary_input_tokens = df['summary_input_tokens'].sum()
        total_summary_output_tokens = df['summary_output_tokens'].sum()
        total_tokens = df['total_tokens'].sum()
        total_cost = df['total_cost_usd'].sum()
        
        print(f"📁 Total Files Analyzed: {total_files}")
        print(f"🔢 Total Tokens: {total_tokens:,}")
        print(f"   - Embedding Tokens: {total_embedding_tokens:,}")
        print(f"   - Summary Input Tokens: {total_summary_input_tokens:,}")
        print(f"   - Summary Output Tokens: {total_summary_output_tokens:,}")
        print(f"💰 Total Cost: ${total_cost:.6f} USD")
        print(f"   - Embedding Cost: ${df['embedding_cost_usd'].sum():.6f} USD")
        print(f"   - Summary Cost: ${df['summary_cost_usd'].sum():.6f} USD")
        
        # 평균 통계
        print(f"\n📈 Average per File:")
        print(f"   - Tokens: {df['total_tokens'].mean():.0f}")
        print(f"   - Cost: ${df['total_cost_usd'].mean():.6f} USD")
        print(f"   - Nodes: {df['total_nodes'].mean():.1f}")
        print(f"   - Layers: {df['num_layers'].mean():.1f}")
        
        # 최대/최소
        max_cost_file = df.loc[df['total_cost_usd'].idxmax()]
        min_cost_file = df.loc[df['total_cost_usd'].idxmin()]
        
        print(f"\n🔝 Highest Cost File:")
        print(f"   - {max_cost_file['filename']}: ${max_cost_file['total_cost_usd']:.6f} USD ({max_cost_file['total_tokens']:,} tokens)")
        
        print(f"🔻 Lowest Cost File:")
        print(f"   - {min_cost_file['filename']}: ${min_cost_file['total_cost_usd']:.6f} USD ({min_cost_file['total_tokens']:,} tokens)")
        
        print("="*80)

    def analyze_single_file(self, pkl_path: str, detailed: bool = True):
        """단일 pkl 파일을 분석하고 상세 정보를 출력"""
        print(f"🔍 Analyzing: {pkl_path}")
        print("-" * 60)
        
        filename, cost_info, tree_info = self.analyze_pkl_file(pkl_path)
        
        if cost_info.total_tokens == 0:
            print("❌ Failed to analyze file or no tokens found")
            return
        
        # 기본 정보 출력
        print(f"📄 File: {filename}")
        print(f"🌳 Tree Structure:")
        print(f"   - Total Nodes: {tree_info.get('total_nodes', 0)}")
        print(f"   - Layers: {tree_info.get('num_layers', 0)}")
        print(f"   - Root Nodes: {tree_info.get('root_nodes', 0)}")
        print(f"   - Leaf Nodes: {tree_info.get('leaf_nodes', 0)}")
        
        print(f"\n🔢 Token Usage:")
        print(f"   - Embedding Tokens: {cost_info.embedding_tokens:,}")
        print(f"   - Summary Input Tokens: {cost_info.summary_input_tokens:,}")
        print(f"   - Summary Output Tokens: {cost_info.summary_output_tokens:,}")
        print(f"   - Total Tokens: {cost_info.total_tokens:,}")
        
        print(f"\n💰 Cost Breakdown (USD):")
        print(f"   - Embedding Cost: ${cost_info.embedding_cost:.6f}")
        print(f"   - Summary Cost: ${cost_info.summary_cost:.6f}")
        print(f"   - Total Cost: ${cost_info.total_cost:.6f}")
        
        if detailed and tree_info.get('layer_distribution'):
            print(f"\n📊 Layer Distribution:")
            for layer, count in tree_info['layer_distribution'].items():
                print(f"   - Layer {layer}: {count} nodes")
        
        print("-" * 60)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze token usage and costs from RAPTOR tree pkl files")
    parser.add_argument("--file", "-f", type=str, help="Analyze a single pkl file")
    parser.add_argument("--directory", "-d", type=str, default="results", 
                       help="Directory containing pkl files (default: results)")
    parser.add_argument("--output", "-o", type=str, default="token_cost_analysis.csv",
                       help="Output CSV file name (default: token_cost_analysis.csv)")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed information for single file analysis")
    
    args = parser.parse_args()
    
    analyzer = TokenCostAnalyzer()
    
    if args.file:
        # 단일 파일 분석
        analyzer.analyze_single_file(args.file, args.detailed)
    else:
        # 모든 pkl 파일 분석
        df = analyzer.analyze_all_pkl_files(args.directory)
        
        if df.empty:
            print(f"No pkl files found in {args.directory}")
            return
        
        # 결과를 CSV로 저장
        df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to: {args.output}")
        
        # 요약 통계 출력
        analyzer.print_summary_statistics(df)
        
        # 상위 10개 파일 출력
        print(f"\n🏆 Top 10 Most Expensive Files:")
        top_10 = df.nlargest(10, 'total_cost_usd')[['filename', 'total_tokens', 'total_cost_usd']]
        for i, row in top_10.iterrows():
            print(f"   {row['filename']}: {row['total_tokens']:,} tokens, ${row['total_cost_usd']:.6f} USD")

if __name__ == "__main__":
    main()