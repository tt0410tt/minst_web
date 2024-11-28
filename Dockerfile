# 1. 기본 이미지 선택
FROM continuumio/miniconda3

# 2. 작업 디렉토리 설정
WORKDIR /app/minst_project

# 3. Python 3.9.20 환경 생성
RUN conda create -n myenv python=3.9.20 -y && \
    conda clean -afy

# 4. Conda 환경 활성화 설정
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# 5. 의존성 설치 (requirements.txt)
COPY requirements.txt .
RUN conda run -n myenv pip install --no-cache-dir -r requirements.txt

# 6. 코드 복사
COPY . .

# 8. 컨테이너 실행 시 서버 실행
CMD ["conda", "run", "-n", "myenv", "python", "code/server/server.py"]
