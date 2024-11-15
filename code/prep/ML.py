import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets"))
model_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "model", "ML"))
os.makedirs(model_dir, exist_ok=True)
result_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "result", "ML"))
os.makedirs(result_dir, exist_ok=True)

# npy 파일 로드
data_file = os.path.join(datasets_dir, "mnist_data.npy")
label_file = os.path.join(datasets_dir, "mnist_labels.npy")
X = np.load(data_file)
y = np.load(label_file, allow_pickle=True)

# 데이터 전처리
if len(X.shape) == 3:
    X = X.reshape(X.shape[0], -1)

# 정규화 (0~1 범위로 스케일링)
X = X / 255.0

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 및 저장 함수
def train_and_save_model(model, model_name):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # 모델 저장
    model_file = os.path.join(model_dir, f"{model_name}_model.npy")
    if hasattr(model, 'coef_'):
        np.save(model_file, model.coef_)
    elif hasattr(model, 'feature_importances_'):
        np.save(model_file, model.feature_importances_)
    else:
        model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
        import joblib
        joblib.dump(model, model_file)
    print(f"{model_name} model saved to: {model_file}")

    # 테스트 결과
    y_pred = model.predict(X_test)

    # 오류 데이터 처리
    incorrect_indices = np.where(y_pred != y_test)[0]
    correct_indices = np.where(y_pred == y_test)[0]

    # 오류 데이터 저장
    error_log_file = os.path.join(result_dir, f"error_{model_name}.txt")
    with open(error_log_file, "w") as log_file:
        log_file.write(f"Total data: {len(X_test)}, Correct: {len(correct_indices)}, Errors: {len(incorrect_indices)}\n")

    errors_dir = os.path.join(result_dir, f"errors_{model_name}")
    os.makedirs(errors_dir, exist_ok=True)
    for idx in incorrect_indices:
        img = X_test[idx].reshape(28, 28)  # 이미지 크기를 복원 (MNIST 기준)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        error_file_name = f"error_{idx}_{y_pred[idx]}_{y_test[idx]}.png"
        error_file_path = os.path.join(errors_dir, error_file_name)
        plt.savefig(error_file_path)
        plt.close()

    print(f"Error images for {model_name} saved to: {errors_dir}")

    # 오류 데이터를 제거한 새로운 npy 파일 저장
    clean_X = X_test[correct_indices]
    clean_y = y_test[correct_indices]
    clean_data_file = os.path.join(datasets_dir, f"clean_data_{model_name}.npy")
    clean_labels_file = os.path.join(datasets_dir, f"clean_labels_{model_name}.npy")
    np.save(clean_data_file, clean_X)
    np.save(clean_labels_file, clean_y)
    print(f"Clean data for {model_name} saved to: {clean_data_file}, {clean_labels_file}")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
train_and_save_model(lr_model, "LogisticRegression")

# Support Vector Machine (SVM)
svm_model = SVC()
train_and_save_model(svm_model, "SVM")

# Naive Bayes
nb_model = GaussianNB()
train_and_save_model(nb_model, "NaiveBayes")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_save_model(rf_model, "RandomForest")
