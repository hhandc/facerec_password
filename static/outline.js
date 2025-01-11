function drawFaceAndShoulderOutline(canvas) {
    const context = canvas.getContext("2d");

    function draw() {
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Set the outline style
        context.strokeStyle = "white";
        context.lineWidth = 4; // Thick line
        context.setLineDash([10, 15]); // Dotted line

        // Begin custom path for the face and shoulders
        context.beginPath();

        // Face (large oval)
        context.ellipse(320, 180, 100, 140, 0, 0, Math.PI * 2); // x, y, radiusX, radiusY, rotation, startAngle, endAngle

        // Left shoulder
        context.moveTo(220, 330);
        context.lineTo(100, 400);
        context.lineTo(60, 470);

        // Right shoulder
        context.moveTo(420, 330);
        context.lineTo(540, 400);
        context.lineTo(580, 470);

        // Stroke the path
        context.stroke();

        // Repeatedly redraw for a live effect
        requestAnimationFrame(draw);
    }

    // Start drawing the outline
    draw();
}