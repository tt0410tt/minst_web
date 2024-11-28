const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");

// Set canvas size
canvas.width = canvas.parentElement.offsetWidth;
canvas.height = canvas.parentElement.offsetHeight;

// Set canvas background to white
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing variables
let drawing = false;

// Canvas event listeners
canvas.addEventListener("mousedown", () => {
    drawing = true;
    console.log("Mouse down");
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
    console.log("Mouse up");
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    console.log("Mouse move", e.clientX, e.clientY);

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.fillStyle = "black";
    ctx.fillRect(x, y, 5, 5);
});

// Recognize button event
document.getElementById("recognize-btn").addEventListener("click", async () => {
    // Check if canvas is empty
    const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    console.log("Canvas pixel data:", pixelData);

    const isCanvasEmpty = pixelData.every((value, index) => {
        // Alpha values are at every 4th index starting from 3
        if ((index + 1) % 4 === 0) {
            return value === 0;
        }
        return true;
    });

    if (isCanvasEmpty) {
        console.error("Canvas is empty. Please draw something.");
        document.getElementById("result-text").textContent = "Canvas is empty!";
        return; // Stop processing if canvas is empty
    }

    // Resize canvas to 28x28 and convert to grayscale
    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    const resizedCtx = resizedCanvas.getContext("2d");

    // Clear resized canvas and draw the resized image
    resizedCtx.clearRect(0, 0, 28, 28);
    resizedCtx.drawImage(
        canvas,
        0,
        0,
        canvas.width,
        canvas.height,
        0,
        0,
        resizedCanvas.width,
        resizedCanvas.height
    );

    // Get image data from resized canvas
    const imageData = resizedCtx.getImageData(0, 0, 28, 28).data;
    console.log("Resized canvas image data:", imageData);

    // Convert image data to grayscale and invert the values
    const grayscaleData = [];
    for (let i = 0; i < imageData.length; i += 4) {
        const gray =
            imageData[i] * 0.3 +   // Red
            imageData[i + 1] * 0.59 + // Green
            imageData[i + 2] * 0.11;  // Blue
        grayscaleData.push(1 - gray / 255.0); // Normalize and invert
    }


    console.log("Grayscale data:", grayscaleData);

    const selectedModel = document.getElementById("model-select").value;

    // Fetch request to send data to the server
    const response = await fetch("/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: grayscaleData, model: selectedModel }),
    });

    const result = await response.json();
    document.getElementById(
        "result-text"
    ).textContent = `Prediction: ${result.prediction}`;
});

// Clear button event
document.getElementById("clear-btn").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Redraw the white background
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result-text").textContent = ""; // Clear result text
});
document.getElementById("model-select").addEventListener("change", async () => {
    // 모델 선택 값을 가져오기
    const selectedModel = document.getElementById("model-select").value;

    if (!selectedModel) {
        alert("No model selected. Please choose a valid model.");
        return; // 선택되지 않았으면 실행 중단
    }

    try {
        // 서버로 요청 전송
        console.log(selectedModel);
        console.log(typeof(selectedModel));
        const response = await fetch("/load_weights", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: selectedModel }), // 모델 이름만 전송
        });

        const result = await response.json();
        if (response.ok) {
            alert(result.message); // 서버 응답 메시지 알림
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        alert(`Failed to load model: ${error.message}`);
    }
});

