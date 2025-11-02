"""Tkinter GUI to draw a digit and have the trained CNN predict it.

Requirements:
- Trained weights + architecture: weights.hdf5 (and weights.json) in project root (created by running example.py)
- Only uses stdlib (tkinter) + numpy + tqdm (already allowed). No Pillow dependency; we maintain our own pixel buffer.

Usage:
    source .venv/bin/activate
    python mnist_gui.py

Draw with left mouse button. Click 'Predict' to see the predicted digit and probabilities.
Click 'Clear' to reset.
"""
import tkinter as tk
import numpy as np
import os
from convnet.model import Model

SCALE = 10          # Base upscale for internal reference (28*10 = 280)
IMG_SIZE = 28
CANVAS_BASE = IMG_SIZE * SCALE  # internal buffer resolution (fixed)
BRUSH_RADIUS_BASE = 4    # in base (CANVAS_BASE) pixels

class DigitGUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        root.title("MNIST Digit Predictor")

        # Layout configuration for responsive resizing
        for col in range(4):
            root.grid_columnconfigure(col, weight=1 if col == 0 else 0)
        root.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, width=CANVAS_BASE, height=CANVAS_BASE, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky='nsew')

        self.predict_btn = tk.Button(root, text='Predict', command=self.predict)
        self.predict_btn.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        self.clear_btn = tk.Button(root, text='Clear', command=self.clear)
        self.clear_btn.grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        self.quit_btn = tk.Button(root, text='Quit', command=root.quit)
        self.quit_btn.grid(row=1, column=2, sticky='ew', padx=2, pady=2)

        self.result_var = tk.StringVar(value='Draw a digit 0-9 then press Predict')
        self.result_label = tk.Label(root, textvariable=self.result_var, font=('Helvetica', 14))
        self.result_label.grid(row=2, column=0, columnspan=4, pady=5)

        # Pixel buffer at fixed base resolution (float32) 0..1 (white is 1.0)
        self.buffer = np.zeros((CANVAS_BASE, CANVAS_BASE), dtype=np.float32)
        self.canvas_w = CANVAS_BASE
        self.canvas_h = CANVAS_BASE

        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.draw)
        self.canvas.bind('<Configure>', self._on_resize)

    def draw(self, event):
        # Map display coords to base buffer coords
        disp_x, disp_y = event.x, event.y
        # Current brush radius scaled relative to base resolution
        scale_x = CANVAS_BASE / self.canvas_w
        scale_y = CANVAS_BASE / self.canvas_h
        base_x = int(disp_x * scale_x)
        base_y = int(disp_y * scale_y)
        r_base = BRUSH_RADIUS_BASE
        # Draw oval on canvas (display radius proportional to resize)
        disp_r = int(max(1, round(r_base / scale_x)))  # approximate uniform
        self.canvas.create_oval(disp_x-disp_r, disp_y-disp_r, disp_x+disp_r, disp_y+disp_r, fill='white', outline='white')
        # Update buffer region
        x0 = max(0, base_x - r_base); x1 = min(CANVAS_BASE, base_x + r_base + 1)
        y0 = max(0, base_y - r_base); y1 = min(CANVAS_BASE, base_y + r_base + 1)
        self.buffer[y0:y1, x0:x1] = 1.0

    def _on_resize(self, event):
        # Scale existing drawings to new size
        if event.width <= 0 or event.height <= 0:
            return
        new_w, new_h = event.width, event.height
        # Maintain square aspect by using min dimension
        side = min(new_w, new_h)
        if side <= 0:
            return
        # Resize canvas (if not already square)
        if (new_w != new_h):
            self.canvas.config(width=side, height=side)
            new_w = new_h = side
        # Compute scale factors
        sx = new_w / self.canvas_w
        sy = new_h / self.canvas_h
        if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3:
            self.canvas.scale('all', 0, 0, sx, sy)
            self.canvas_w = new_w
            self.canvas_h = new_h

    def clear(self):
        self.canvas.delete('all')
        self.buffer.fill(0.0)
        self.result_var.set('Cleared. Draw again.')

    def _prepare_image(self):
        # Downscale 280x280 -> 28x28 by average pooling each 10x10 block
        small = self.buffer.reshape(IMG_SIZE, SCALE, IMG_SIZE, SCALE).mean(axis=(1,3))
        img = small.astype(np.float32)
        # Threshold to reduce noise
        img[img < 0.2] = 0.0
        # Normalize max to 1
        if img.max() > 0:
            img /= img.max()
        # Center digit via centroid shift
        coords = np.argwhere(img > 0)
        if coords.size > 0:
            cy, cx = coords.mean(axis=0)
            shift_y = int(round(IMG_SIZE/2 - cy))
            shift_x = int(round(IMG_SIZE/2 - cx))
            img = np.roll(img, shift_y, axis=0)
            img = np.roll(img, shift_x, axis=1)
        # Optional: light Gaussian blur (manual 3x3) to approximate MNIST stroke softness
        if coords.size > 0:
            k = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)
            k /= k.sum()
            padded = np.pad(img, 1)
            blurred = np.zeros_like(img)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    patch = padded[i:i+3, j:j+3]
                    blurred[i,j] = (patch * k).sum()
            img = blurred
            if img.max() > 0:
                img /= img.max()
        return img[None, ..., None]

    def predict(self):
        img = self._prepare_image()
        # Ensure model built
        if not self.model.built:
            self.model.build(img.shape)
        logits = self.model.predict(img)
        # Softmax manually (final layer has no activation)
        e = np.exp(logits - logits.max())
        probs = (e / e.sum()).ravel()
        pred = int(probs.argmax())
        # Format top 3
        top3_idx = probs.argsort()[-3:][::-1]
        top_str = ', '.join(f"{i}:{probs[i]*100:.1f}%" for i in top3_idx)
        self.result_var.set(f"Pred: {pred} | {top_str}")


def load_model(weights_path='weights.hdf5'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' not found. Run example.py to train and create it.")
    model = Model.load(weights_path)
    # Build with dummy input to finalize shapes
    model.build((1, IMG_SIZE, IMG_SIZE, 1))
    return model


def main():
    model = load_model()
    root = tk.Tk()
    gui = DigitGUI(root, model)
    root.mainloop()

if __name__ == '__main__':
    main()
