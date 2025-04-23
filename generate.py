import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from model import HandwritingGenerator

def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = HandwritingGenerator(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_handwriting(model, start_seq, length=100):
    model.eval()
    generated = [start_seq]
    input_seq = torch.tensor(start_seq, dtype=torch.float32).unsqueeze(0)
    hidden = None
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            next_point = output[:, -1, :].squeeze(0).numpy()
            generated.append(next_point)
            input_seq = torch.tensor(next_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return np.array(generated)

def plot_and_save_strokes(strokes, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(8, 2))
    x, y = 0, 0
    X, Y = [x], [y]
    for i in range(1, len(strokes)):
        dx, dy, pen = strokes[i]
        x += dx
        y += dy
        X.append(x)
        Y.append(y)
        if pen < 0.5:
            plt.plot(X, Y, 'k-', linewidth=2)
            X, Y = [x], [y]
    plt.axis('off')
    timestamp = int(time.time())
    filename = os.path.join(output_dir, f'{timestamp}.jpeg')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Handwritten output saved as {filename}')

def text_to_stroke_sequence(text):
    # Placeholder: Replace with actual text-to-stroke conversion logic
    # For now, use a random starting sequence
    return np.zeros((10, 3))

def main():
    # Model parameters (should match training)
    input_size = 3
    hidden_size = 256
    num_layers = 2
    output_size = 3
    model_path = 'output/handwriting_model.pth'

    model = load_model(model_path, input_size, hidden_size, num_layers, output_size)
    text = input('Enter text to generate as handwriting: ')
    start_seq = text_to_stroke_sequence(text)
    generated_strokes = generate_handwriting(model, start_seq)
    plot_and_save_strokes(generated_strokes)

if __name__ == '__main__':
    main()
