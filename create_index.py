#!/usr/bin/env python3
"""
편의 실행 스크립트 - Azure Search 인덱스 생성
"""

import sys
import os

# 루트 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 실제 스크립트 import 및 실행
if __name__ == "__main__":
    from scripts.azure.create_optimized_raptor import main

    main()
