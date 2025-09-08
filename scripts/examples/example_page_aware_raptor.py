#!/usr/bin/env python3
"""
페이지 인식 RAPTOR 사용 예시

약관 데이터에서 페이지 정보를 활용한 클러스터링을 수행합니다.
페이지 간격이 큰 내용들은 클러스터링하지 않아 논리적 일관성을 유지합니다.
"""

import pandas as pd
from config.page_aware_config import create_page_aware_builder


def prepare_chunked_data_with_pages(df):
    """
    DataFrame에서 페이지 정보를 포함한 청크 데이터 준비

    Args:
        df: page_number와 content 컬럼을 가진 DataFrame

    Returns:
        list: 페이지 정보가 포함된 청크 리스트
    """
    chunks = []

    for idx, row in df.iterrows():
        chunk = {"text": row.get("content", ""), "page_number": row.get("page_number")}
        chunks.append(chunk)

    return chunks


def process_with_page_aware_raptor(csv_file, max_page_gap=10):
    """
    페이지 인식 RAPTOR로 약관 문서 처리

    Args:
        csv_file: 입력 CSV 파일 경로
        max_page_gap: 클러스터링을 허용할 최대 페이지 간격
    """
    print(f"📄 파일 로딩: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"📊 데이터 현황:")
    print(f"  - 총 행 수: {len(df)}")
    if "page_number" in df.columns:
        page_range = df["page_number"].agg(["min", "max"])
        print(f"  - 페이지 범위: {page_range['min']} ~ {page_range['max']}")
        print(f"  - 고유 페이지 수: {df['page_number'].nunique()}")
    else:
        print("  - ⚠️ page_number 컬럼이 없습니다")

    # 페이지 인식 RAPTOR 빌더 생성
    print(f"\n🌳 페이지 인식 RAPTOR 빌더 생성 (최대 페이지 간격: {max_page_gap})")
    builder = create_page_aware_builder(max_page_gap=max_page_gap)

    # source_title별로 그룹핑하여 처리
    if "source_title" in df.columns:
        source_titles = df["source_title"].unique()
        print(f"📋 처리할 상품: {len(source_titles)}개")

        for source_title in source_titles:
            print(f"\n🔄 처리 중: {source_title}")

            # 해당 상품의 데이터만 필터링
            product_df = df[df["source_title"] == source_title].copy()

            # 페이지 번호로 정렬
            if "page_number" in product_df.columns:
                product_df = product_df.sort_values("page_number")

            # 청크 데이터 준비
            chunks = prepare_chunked_data_with_pages(product_df)

            print(f"  📄 청크 수: {len(chunks)}")
            if "page_number" in product_df.columns:
                page_info = product_df["page_number"].agg(["min", "max", "nunique"])
                print(
                    f"  📄 페이지: {page_info['min']}~{page_info['max']} ({page_info['nunique']}개)"
                )

            # RAPTOR 트리 구축
            print(f"  🌳 RAPTOR 트리 구축 시작...")
            tree = builder.build_from_text(
                text="",  # 텍스트는 청크에서 가져옴
                chunked_list=chunks,
                use_multithreading=True,
            )

            print(
                f"  ✅ 완료 - 총 노드: {len(tree.all_nodes)}, 레이어: {tree.num_layers}"
            )

            # 페이지별 노드 분포 확인
            page_distribution = {}
            for node in tree.all_nodes.values():
                if hasattr(node, "page_number") and node.page_number is not None:
                    page_num = node.page_number
                    if page_num not in page_distribution:
                        page_distribution[page_num] = 0
                    page_distribution[page_num] += 1

            if page_distribution:
                print(f"  📊 페이지별 노드 분포 (상위 10개):")
                sorted_pages = sorted(page_distribution.items())[:10]
                for page, count in sorted_pages:
                    print(f"    페이지 {page}: {count}개 노드")

    else:
        print("⚠️ source_title 컬럼이 없어 전체 데이터를 하나로 처리합니다")

        # 전체 데이터 처리
        chunks = prepare_chunked_data_with_pages(df)
        tree = builder.build_from_text(
            text="", chunked_list=chunks, use_multithreading=True
        )

        print(f"✅ 완료 - 총 노드: {len(tree.all_nodes)}, 레이어: {tree.num_layers}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="페이지 인식 RAPTOR 처리")
    parser.add_argument("csv_file", help="입력 CSV 파일 경로")
    parser.add_argument(
        "--max-page-gap",
        type=int,
        default=10,
        help="클러스터링을 허용할 최대 페이지 간격 (기본: 10)",
    )

    args = parser.parse_args()

    print("🔍 페이지 인식 RAPTOR 처리 시작")
    print(f"📁 입력 파일: {args.csv_file}")
    print(f"📏 최대 페이지 간격: {args.max_page_gap}")
    print("-" * 60)

    try:
        process_with_page_aware_raptor(args.csv_file, args.max_page_gap)
        print("\n🎉 처리 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
