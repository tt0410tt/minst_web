const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Resize the canvas to match the container
canvas.width = canvas.parentElement.offsetWidth;
canvas.height = canvas.parentElement.offsetHeight;

// Start drawing
canvas.addEventListener('mousedown', () => {
    drawing = true;
    ctx.beginPath();
});

// Draw on the canvas
canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        ctx.lineTo(x, y);
        ctx.stroke();
    }
});

// Stop drawing
canvas.addEventListener('mouseup', () => {
    drawing = false;
});

// Handle the "인식" button click
document.getElementById('recognize-btn').addEventListener('click', () => {
    const selectedModel = document.getElementById('model-select').value;

    console.log(`Selected Model: ${selectedModel}`);

    fetch('/recognize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ selectedModel }),
    })
        .then((response) => response.json())
        .then((data) => {
            alert(data.message);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});

// Handle the "지우기" button click
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});
