#!/usr/bin/env python3
"""
results/recursive 폴더의 모든 pkl 파일을 불러와서
character_combined_raptor_data.csv와 동일한 형식으로 하나의 CSV 파일에 저장하는 스크립트
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json


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
    """
    RAPTOR Tree 객체에서 CSV 형식에 맞는 정보 추출
    Tree 객체는 all_nodes, layer_to_nodes 등의 속성을 가짐
    """
    rows = []

    # Tree 객체인지 확인
    if hasattr(data, "all_nodes") and hasattr(data, "layer_to_nodes"):
        # all_nodes에서 모든 노드 정보 추출
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

    return rows


def extract_node_info(node, source_title, node_to_layer):
    """개별 Node 객체에서 정보 추출"""
    try:
        # Node 객체에서 정보 추출
        content = getattr(node, "text", "")
        node_index = getattr(node, "index", 0)
        children = getattr(node, "children", set())

        # 임베딩 벡터 추출 (embeddings 딕셔너리에서 'EMB' 키)
        embeddings_dict = getattr(node, "embeddings", {})
        vector = embeddings_dict.get("EMB", []) if embeddings_dict else []

        # 레이어 정보
        layer = node_to_layer.get(node_index, 0)

        # 루트/리프 노드 판별
        is_leaf = len(children) == 0
        is_root = layer == max(node_to_layer.values()) if node_to_layer else False

        # 벡터를 문자열로 변환 (CSV 저장용)
        if isinstance(vector, (list, tuple, np.ndarray)):
            if len(vector) > 0:
                vector_str = json.dumps(
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
            else:
                vector_str = "[]"
        else:
            vector_str = "[]"

        # children을 문자열로 변환 (set을 list로 변환)
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


def main():
    """메인 함수"""
    terms = "iterms"  # "eterms" or "iterms"
    chunck = "basic"  # "basic" or "table"
    results_dir = Path(f"results/{terms}/{chunck}")
    output_file = f"results/{terms}_{chunck}_combined_raptor_data.csv"

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist!")
        return

    # 모든 pkl 파일 찾기
    pkl_files = list(results_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pkl files")

    all_rows = []

    # 각 pkl 파일 처리
    for pkl_file in pkl_files:
        print(f"Processing {pkl_file.name}...")

        # 파일명에서 source_title 추출 (확장자 제거)
        source_title = pkl_file.stem

        # pkl 파일 로드
        data = load_pkl_file(pkl_file)
        if data is None:
            continue

        # 데이터 추출
        rows = extract_raptor_data(data, source_title)
        all_rows.extend(rows)
        print(f"  Extracted {len(rows)} rows")

    if not all_rows:
        print("No data extracted from pkl files!")
        return

    # DataFrame 생성
    df = pd.DataFrame(all_rows)

    # 컬럼 순서 맞추기
    columns = [
        "source_title",
        "node_index",
        "ko_content",
        "ko_content_vector",
        "children",
        "layer",
        "is_root",
        "is_leaf",
    ]
    df = df[columns]

    # CSV 저장
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows to {output_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # 샘플 데이터 출력
    print("\nSample data:")
    print(df.head(3))


if __name__ == "__main__":
    main()
