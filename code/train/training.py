import os
import numpy as np
from sklearn.model_selection import train_test_split

class Model:
    learning_rate = 0.1
    input_size = 784
    hidden_size1 = 50
    hidden_size2 = 100
    output_size = 10

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        return exp_a / sum_exp_a

    def __init__(self):
        # Initialize weights and biases using a normal distribution
        self.W1 = np.random.normal(0, np.sqrt(2 / self.input_size), (self.input_size, self.hidden_size1))
        self.W2 = np.random.normal(0, np.sqrt(2 / self.hidden_size1), (self.hidden_size1, self.hidden_size2))
        self.W3 = np.random.normal(0, np.sqrt(2 / self.hidden_size2), (self.hidden_size2, self.output_size))
        self.b1 = np.random.normal(0, np.sqrt(2 / self.input_size), self.hidden_size1)
        self.b2 = np.random.normal(0, np.sqrt(2 / self.hidden_size1), self.hidden_size2)
        self.b3 = np.random.normal(0, np.sqrt(2 / self.hidden_size2), self.output_size)

    def predict(self, x_batch):
        a1 = np.dot(x_batch, self.W1) + self.b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        return self.softmax(a3)

    def backpropagation(self, x_batch, y_batch, batch_size):
        a1 = np.dot(x_batch, self.W1) + self.b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        y_pred = self.softmax(a3)

        delta3 = (y_pred - y_batch) / batch_size
        dW3 = np.dot(z2.T, delta3)
        db3 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.W3.T) * z2 * (1 - z2)
        dW2 = np.dot(z1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, self.W2.T) * z1 * (1 - z1)
        dW1 = np.dot(x_batch.T, delta1)
        db1 = np.sum(delta1, axis=0)

        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

# 데이터 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets"))
model_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "model", "MyDL"))
result_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "result", "MyDL"))
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 데이터 로드 및 분할
def load_data(clean_data_file, clean_labels_file):
    X = np.load(os.path.join(datasets_dir, clean_data_file), allow_pickle=True)
    y = np.load(os.path.join(datasets_dir, clean_labels_file), allow_pickle=True)
    if len(y.shape) > 1 and y.shape[1] > 1:
        return X, y  # Already one-hot encoded
    else:
        y = y.astype(int)
        y_one_hot = np.eye(10)[y]
        return X, y_one_hot

# 학습 및 저장
def train(model_name, clean_data_file, clean_labels_file):
    X_train, y_train = load_data(clean_data_file, clean_labels_file)
    model = Model()

    epochs = 200
    batch_size = 32
    high_accuracy_count = 0
    log_file_path = os.path.join(result_dir, f"result_{model_name}.txt")

    with open(log_file_path, "w") as log_file:
        for epoch in range(epochs):
            total_loss = 0
            accuracy = 0

            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                model.backpropagation(x_batch, y_batch, batch_size)

                y_pred = model.predict(x_batch)
                loss = np.mean((y_batch - y_pred) ** 2)
                total_loss += loss

                accuracy += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

            accuracy = accuracy / len(X_train)
            log_message = f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}\n"
            print(log_message, end="")
            log_file.write(log_message)

            if accuracy > 0.9995:
                high_accuracy_count += 1
                if high_accuracy_count >= 5:
                    print(f"Stopping early as accuracy > 0.9995 for 5 consecutive epochs.\n")
                    log_file.write("Stopping early as accuracy > 0.9995 for 5 consecutive epochs.\n")
                    break
            else:
                high_accuracy_count = 0

    # 모델 저장
    np.save(os.path.join(model_dir, f"W1_{model_name}.npy"), model.W1)
    np.save(os.path.join(model_dir, f"W2_{model_name}.npy"), model.W2)
    np.save(os.path.join(model_dir, f"W3_{model_name}.npy"), model.W3)
    np.save(os.path.join(model_dir, f"b1_{model_name}.npy"), model.b1)
    np.save(os.path.join(model_dir, f"b2_{model_name}.npy"), model.b2)
    np.save(os.path.join(model_dir, f"b3_{model_name}.npy"), model.b3)
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    datasets = [
        ("Original", "mnist_data.npy", "mnist_labels.npy"),
        ("LogisticRegression", "clean_data_LogisticRegression.npy", "clean_labels_LogisticRegression.npy"),
        ("SVM", "clean_data_SVM.npy", "clean_labels_SVM.npy"),
        ("NaiveBayes", "clean_data_NaiveBayes.npy", "clean_labels_NaiveBayes.npy"),
        ("RandomForest", "clean_data_RandomForest.npy", "clean_labels_RandomForest.npy")
    ]

    for model_name, data_file, labels_file in datasets:
        train(model_name, data_file, labels_file)
