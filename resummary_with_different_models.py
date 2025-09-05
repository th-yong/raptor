#!/usr/bin/env python3
"""
기존 pkl 파일에서 클러스터링은 그대로 유지하고 서머리만 다른 LLM 모델로 재생성하는 스크립트
"""

import pickle
import os
import logging
from typing import Dict, List
from pathlib import Path

from dotenv import load_dotenv
from raptor.tree_structures import Tree, Node
from raptor.SummarizationModels import AzureSummarizationModel
from utils.llm_manager import AzureAIClientManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TreeResummarizer:
    """기존 트리에서 서머리만 다른 모델로 재생성하는 클래스"""
    
    def __init__(self, new_model_deployment: str):
        """
        Args:
            new_model_deployment: 새로 사용할 모델 deployment 이름 (gpt-4o, gpt-4o-mini, gpt-4.1-mini)
        """
        self.new_model_deployment = new_model_deployment
        
        # Azure 클라이언트 초기화
        self.client = AzureAIClientManager(deployment=new_model_deployment)
        self.summarization_model = AzureSummarizationModel(self.client)
        
        logger.info(f"TreeResummarizer initialized with model: {new_model_deployment}")
    
    def load_tree(self, pkl_path: str) -> Tree:
        """pkl 파일에서 트리 로드"""
        with open(pkl_path, 'rb') as f:
            tree = pickle.load(f)
        logger.info(f"Loaded tree from {pkl_path}")
        return tree
    
    def save_tree(self, tree: Tree, output_path: str):
        """트리를 pkl 파일로 저장"""
        with open(output_path, 'wb') as f:
            pickle.dump(tree, f)
        logger.info(f"Saved tree to {output_path}")
    
    def identify_summary_nodes(self, tree: Tree) -> List[Node]:
        """서머리가 필요한 노드들을 식별 (리프 노드가 아닌 노드들)"""
        summary_nodes = []
        
        for node_id, node in tree.all_nodes.items():
            # 자식이 있는 노드 = 서머리 노드
            if len(node.children) > 0:
                summary_nodes.append(node)
        
        logger.info(f"Found {len(summary_nodes)} summary nodes to re-summarize")
        return summary_nodes
    
    def get_children_text(self, node: Node, tree: Tree) -> str:
        """노드의 자식들의 텍스트를 합쳐서 반환"""
        children_texts = []
        
        for child_id in node.children:
            if child_id in tree.all_nodes:
                children_texts.append(tree.all_nodes[child_id].text)
        
        return "\n\n".join(children_texts)
    
    def resummarize_node(self, node: Node, tree: Tree) -> str:
        """특정 노드를 새로운 모델로 재서머리"""
        # 자식 노드들의 텍스트 수집
        children_text = self.get_children_text(node, tree)
        
        # 새로운 모델로 서머리 생성
        new_summary = self.summarization_model.summarize(
            context=children_text,
            max_tokens=500
        )
        
        return new_summary
    
    def resummarize_tree(self, tree: Tree) -> Tree:
        """전체 트리의 서머리 노드들을 재생성"""
        # 서머리 노드들 식별
        summary_nodes = self.identify_summary_nodes(tree)
        
        logger.info(f"Starting re-summarization with {self.new_model_deployment}")
        
        # 각 서머리 노드를 새로운 모델로 재생성
        for i, node in enumerate(summary_nodes):
            try:
                logger.info(f"Re-summarizing node {node.index} ({i+1}/{len(summary_nodes)})")
                
                # 새로운 서머리 생성
                new_summary = self.resummarize_node(node, tree)
                
                # 노드의 텍스트 업데이트 (임베딩과 클러스터링은 그대로 유지)
                node.text = new_summary
                
                logger.info(f"Successfully re-summarized node {node.index}")
                
            except Exception as e:
                logger.error(f"Failed to re-summarize node {node.index}: {e}")
                # 실패한 경우 원본 텍스트 유지
                continue
        
        logger.info("Re-summarization completed")
        return tree


def main():
    """메인 실행 함수"""
    
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    # Azure 환경변수 확인
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        logger.error("AZURE_OPENAI_ENDPOINT와 AZURE_OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        logger.error(".env 파일을 확인하거나 환경변수를 설정해주세요.")
        return
    
    # 테스트할 모델들
    models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]
    
    # 입력 pkl 파일 경로 (예시)
    input_pkl_path = "./results/eterms/basic_1file_test/11384 11386 무배당 심뇌혈관종합건강보험(무해지환급금형).pkl"
    
    # 파일이 존재하는지 확인
    if not os.path.exists(input_pkl_path):
        logger.error(f"Input file not found: {input_pkl_path}")
        return
    
    # 각 모델로 재서머리 수행
    for model in models_to_test:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing with model: {model}")
            logger.info(f"{'='*50}")
            
            # TreeResummarizer 초기화
            resummarizer = TreeResummarizer(model)
            
            # 원본 트리 로드
            original_tree = resummarizer.load_tree(input_pkl_path)
            
            # 재서머리 수행
            new_tree = resummarizer.resummarize_tree(original_tree)
            
            # 결과 저장
            base_name = Path(input_pkl_path).stem
            output_path = f"results/resummarized_{base_name}_{model.replace('.', '_')}.pkl"
            resummarizer.save_tree(new_tree, output_path)
            
            logger.info(f"Completed re-summarization with {model}")
            logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process with model {model}: {e}")
            continue


if __name__ == "__main__":
    main()