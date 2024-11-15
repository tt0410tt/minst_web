import os
from sklearn.datasets import fetch_openml
import numpy as np

# MNIST 데이터셋 다운로드
print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# 데이터 확인
print(f"Shape of data: {X.shape}, Shape of labels: {y.shape}")

# 스크립트 파일 위치 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 디렉토리로 이동
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
# 'datasets' 디렉토리 진입
datasets_dir = os.path.join(parent_dir, "datasets")
os.makedirs(datasets_dir, exist_ok=True)

# 데이터 저장
np.save(os.path.join(datasets_dir, "mnist_data.npy"), X)
np.save(os.path.join(datasets_dir, "mnist_labels.npy"), y)

print(f"MNIST dataset saved in: {datasets_dir}")
