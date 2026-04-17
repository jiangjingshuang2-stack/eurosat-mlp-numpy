# EuroSAT MLP Classifier from Scratch

This repository contains a NumPy implementation of a three-layer MLP classifier for EuroSAT RGB land-cover image classification. The project does not use PyTorch, TensorFlow, JAX, or other automatic differentiation frameworks.

## Project Structure

```text
main/
├── data.py          # dataset loading, preprocessing, split, mini-batch iterator
├── layers.py        # Linear, ReLU, Sigmoid, Tanh layers and backward propagation
├── loss.py          # softmax cross-entropy loss
├── model.py         # three-layer MLP model
├── optimizer.py     # SGD, weight decay, learning-rate decay
├── train.py         # training loop, validation, checkpointing, training curves
├── test.py          # test accuracy, confusion matrix, error analysis examples
├── search.py        # grid search for hyperparameters
├── visualize.py     # first-layer weight visualization
├── plot_confusion_matrix.py # save confusion_matrix.npy as a PNG figure
├── requirements.txt
└── README.md
```

The expected dataset layout is:

```text
hw1/
├── EuroSAT_RGB/
│   ├── AnnualCrop/
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   ├── Highway/
│   ├── Industrial/
│   ├── Pasture/
│   ├── PermanentCrop/
│   ├── Residential/
│   ├── River/
│   └── SeaLake/
└── main/
    └── ...
```

All scripts resolve paths relative to the `main/` directory, so they can be run either from `main/` or from the parent `hw1/` directory.

## Environment

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Required packages:

```text
numpy
Pillow
matplotlib
```

## Data Check

Run:

```bash
python3 data.py
```

This prints dataset classes, total data shape, train/validation/test split shapes, and a sample mini-batch shape.

## Training

Run:

```bash
python3 train.py
```

The training script:

- loads EuroSAT RGB images and resizes them to `32 x 32`;
- flattens each image into a `3072`-dimensional vector;
- normalizes pixels to `[0, 1]`;
- standardizes train/validation/test features using training-set mean and standard deviation;
- trains a three-layer MLP;
- applies softmax cross-entropy, SGD, L2 weight decay, and learning-rate decay;
- saves the best model according to validation accuracy.

Main default hyperparameters:

```text
hidden_dim1 = 256
hidden_dim2 = 128
activation = relu
epochs = 100
batch_size = 64
lr = 0.01
weight_decay = 1e-3
lr_decay = 0.95
```

Training outputs are saved to:

```text
outputs/checkpoints/best_model.npz
outputs/logs/train_history.json
outputs/curves/loss_curve.png
outputs/curves/val_accuracy_curve.png
```

Final result from the 100-epoch run:

```text
Best validation accuracy: 0.6741
Test loss: 0.9869
Test accuracy: 0.6588
```

For long server runs:

```bash
nohup python3 train.py > train.log 2>&1 &
tail -f train.log
```

## Testing

After training, run:

```bash
python3 test.py
```

The test script loads `outputs/checkpoints/best_model.npz`, evaluates the independent test set, prints accuracy, prints the confusion matrix, and saves:

```text
outputs/confusion_matrix.npy
outputs/error_analysis/misclassified_*.png
```

The error-analysis images are saved from the original normalized `[0, 1]` images, while model evaluation uses standardized inputs.

To save the confusion matrix as a PNG figure for the report, run:

```bash
python3 plot_confusion_matrix.py
```

This saves:

```text
outputs/confusion_matrix.png
```

## Hyperparameter Search

Run:

```bash
python3 search.py
```

The search script performs grid search over hidden dimensions, activation function, learning rate, and weight decay. Results are saved to:

```text
outputs/search/search_results_partial.json
outputs/search/search_results.json
outputs/search/best_config.json
```

Use the final search table in the experiment report to satisfy the hyperparameter-search requirement.

## Weight Visualization

After training, run:

```bash
python3 visualize.py
```

This restores first-layer weights to image shape and saves visualizations to:

```text
outputs/weights_vis/first_layer_weights.png
outputs/weights_vis/single_neurons/
```

These figures can be used in the report to discuss whether the first layer captures color or spatial texture patterns.

## Model Weights

The trained model checkpoint is saved at:

```text
outputs/checkpoints/best_model.npz
```

If the checkpoint is too large for GitHub upload, upload it to Google Drive or another file host and include the download link in the final report.

## Report Checklist

The final PDF report should include:

- model structure and implementation details;
- data loading, preprocessing, and train/validation/test split;
- training and validation loss curves;
- validation accuracy curve;
- hyperparameter-search results;
- final test accuracy;
- confusion matrix;
- first-layer weight visualization and discussion;
- several misclassified examples and error analysis;
- GitHub repository link;
- trained model checkpoint download link.
