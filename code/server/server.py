from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    selected_model = data.get('selectedModel')
    # 여기에서 선택된 모델을 처리하는 로직 추가
    return jsonify({'message': f'Recognition triggered with model: {selected_model}!'})

if __name__ == '__main__':
    app.run(debug=True)
