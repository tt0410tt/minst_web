from flask import Flask, request, jsonify, render_template
import numpy as np
import sys
import os

# 현재 디렉토리의 상위 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train.training import Model  # Model 클래스 가져오기

app = Flask(__name__)

# Model 클래스 초기화
model = Model()

@app.route("/")
def index():
    """
    첫 페이지를 렌더링합니다.
    """
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    """
    사용자가 업로드한 데이터를 통해 모델의 예측을 수행합니다.
    """
    try:
        # 클라이언트에서 보낸 데이터 가져오기
        data = request.json

        input_data = np.array(data.get("data")).reshape(1, 784)  # 28x28 데이터를 1x784로 변환
        print(data)
        # 모델 예측 수행
        predictions = model.predict(input_data)
        predicted_class = int(np.argmax(predictions))  # 가장 높은 확률의 클래스

        # 결과 반환
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/load_weights", methods=["POST"])
def load_weights():
    """
    클라이언트가 요청한 가중치 파일을 모델에 로드합니다.
    """
    try:
        # 요청에서 가중치 디렉토리 경로 가져오기
        data = request.json
        print(data['model'])
        
        # 모델에 가중치 로드
        model.load_weights(data['model'])
        return jsonify({"message": f"Weights loaded successfully from {data}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
