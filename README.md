# Handwritten Text Generation

This project generates realistic handwritten-style text using a deep learning model (LSTM-based) trained on the DeepWriting dataset.

## Objectives
- Generate realistic handwritten-style text using a custom-trained neural network.
- Provide a simple interface to train, validate, and generate handwritten text images from user input.

## Project Structure
- `data/` — Contains the training and validation datasets (`deepwriting_training.npz`, `deepwriting_validation.npz`).
- `output/` — Stores the trained model and generated handwritten images.
- `model.py` — Defines the LSTM-based model architecture.
- `process.ipynb` — Jupyter notebook for data preprocessing, model training, and validation.
- `generate.py` — Script to generate handwritten text images from user input using the trained model.
- `PROJECT_FLOW.md` — Project workflow and planning document.
- `README.md` — Project documentation and instructions.

## Step-by-Step Tutorial

### 1. Prerequisites
- Python 3.7+
- Install required packages:
  ```bash
  pip install torch numpy matplotlib
  ```

### 2. Prepare the Dataset
- Ensure the following files are present in the `data/` directory:
  - `deepwriting_training.npz`
  - `deepwriting_validation.npz`

### 3. Train the Model
- Open and run all cells in `process.ipynb` to:
  - Load and preprocess the data
  - Train the model
  - Validate and visualize training progress
  - Save the trained model to `output/handwriting_model.pth`

### 4. Generate Handwritten Text
- Run the following command:
  ```bash
  python generate.py
  ```
- Enter the text you want to generate when prompted.
- The generated handwritten image will be saved in the `output/` directory as `<timestamp>.jpeg`.

## Hyperparameter Tuning
- You can adjust model parameters (hidden size, number of layers, etc.) in `model.py` and in the notebook `process.ipynb`.

## Notes
- The text-to-stroke conversion in `generate.py` is a placeholder. For best results, implement a mapping from text to initial stroke sequences.
- All code is modular and well-commented for clarity.

## License
This project is for educational and research purposes.
