
# Meat Freshness Classification Project

## Overview
This project is designed to classify images of meat into three categories: Fresh, Half-Fresh, and Spoiled. It uses a deep learning model built on PyTorch with a pre-trained ResNet18 architecture that is fine-tuned for this specific task.

## Prerequisites
- Python 3.6 or later
- PyTorch
- Torchvision
- Pandas
- Pillow
- Matplotlib

## Installation
1. Clone this repository:
   ```bash
   git clone https://your-repository-url.git
   cd Meat-Freshness-Classification
   ```
2. Install required packages:
   ```bash
   pip install torch torchvision pandas pillow matplotlib
   ```

## Dataset
The dataset should be structured in the following directory format:
```
meat2/
├── train/
│   ├── _classes.csv
│   └── images/
└── valid/
    ├── _classes.csv
    └── images/
```
- `_classes.csv` should contain the image filenames and their corresponding labels.

## Usage
1. Load and preprocess the data:
   Adjust paths in the script if your dataset directory structure differs.

2. Train the model:
   Run the sections of the script sequentially to train the model using the training dataset.

3. Validate the model:
   After training, run the model against the validation set to evaluate its performance.

4. Analysis:
   Generate a classification report and confusion matrix to understand the model's performance across the classes.

## Model Training
Modify the training parameters, such as `num_epochs` and `batch_size`, to experiment with different training configurations.

## Saving and Loading the Model
The model weights are saved to `model_weights.pth`. You can modify the script to change the path or filename as necessary.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your features or enhancements.

## License
 MIT License.

