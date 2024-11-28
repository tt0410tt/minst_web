import os
import numpy as np

def inspect_npy_file():
    # 현재 스크립트 위치에서 상위 경로로 이동하여 파일 경로 설정
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "mnist_data.npy"))
    
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return
    
    try:
        # .npy 파일 로드
        data = np.load(file_path)
        
        # 데이터 정보 출력
        print(f"Loaded file: {file_path}")
        print(f"Data type: {type(data)}")
        print(f"Array shape: {data.shape}")
        print(f"Data type of elements: {data.dtype}")
        
        # 첫 번째 28x28 데이터 출력
        print("\nFirst 28x28 data (as a 2D array):")
        print(data[0])  # 첫 번째 데이터를 출력
        
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

# 실행
if __name__ == "__main__":
    inspect_npy_file()
