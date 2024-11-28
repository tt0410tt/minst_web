const gridContainer = document.getElementById("drawing-grid");
const gridSize = 28; // 28x28 grid
const grid = [];
let isDragging = false; // 드래그 상태를 추적

// 28x28 그리드 생성
for (let i = 0; i < gridSize * gridSize; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";

    // 마우스 드래그 시작 시 색상 설정
    cell.addEventListener("mousedown", () => {
        isDragging = true;
        setCellBlack(cell);
    });

    // 드래그 중 색상 설정
    cell.addEventListener("mousemove", () => {
        if (isDragging) {
            setCellBlack(cell);
        }
    });

    gridContainer.appendChild(cell);
    grid.push(cell);
}

// 마우스 버튼을 놓으면 드래그 중단
document.body.addEventListener("mouseup", () => {
    isDragging = false;
});

// 셀을 검정색으로 설정하는 함수
function setCellBlack(cell) {
    cell.style.backgroundColor = "black";
}

// Recognize 버튼 클릭 이벤트
document.getElementById("recognize-btn").addEventListener("click", async () => {
    // 그리드 상태를 데이터로 변환 (검정: 1, 흰색: 0)
    const grayscaleData = grid.map((cell) => (cell.style.backgroundColor === "black" ? 1 : 0));
    console.log("Grayscale data:", grayscaleData);

    const selectedModel = document.getElementById("model-select").value;

    try {
        const response = await fetch("/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: grayscaleData, model: selectedModel }),
        });

        const result = await response.json();
        document.getElementById("result-text").textContent = `Prediction: ${result.prediction}`;
    } catch (error) {
        console.error("Error during recognition:", error);
    }
});

// Clear 버튼 클릭 이벤트
document.getElementById("clear-btn").addEventListener("click", () => {
    grid.forEach((cell) => (cell.style.backgroundColor = "white"));
    document.getElementById("result-text").textContent = ""; // 결과 초기화
});
// 가중치 설정 함수
async function loadModelWeights(selectedModel) {
    try {
        // 서버로 선택한 모델 전송
        const response = await fetch("/load_weights", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: selectedModel }), // 선택된 모델 이름 전송
        });

        const result = await response.json();
        if (response.ok) {
            console.log(`Model weights loaded: ${result.message}`);
            alert(`Model weights for '${selectedModel}' loaded successfully.`);
        } else {
            console.error(`Failed to load model weights: ${result.error}`);
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error(`Error while loading model weights: ${error}`);
        alert(`Failed to load model weights: ${error.message}`);
    }
}

// 모델 선택 드롭다운 이벤트 리스너
document.getElementById("model-select").addEventListener("change", (event) => {
    const selectedModel = event.target.value;
    if (selectedModel) {
        console.log(`Selected model: ${selectedModel}`);
        loadModelWeights(selectedModel); // 가중치 로드 호출
    }
});
