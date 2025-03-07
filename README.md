# Deep Learning Multi-Model Project

This repository contains a Jupyter Notebook for a deep learning project focused on multi-lingual sentiment analysis using various tools and libraries for data processing, model training, and evaluation.

## Project Overview

### Dependencies
The following libraries are used in this project:
- `numpy`
- `pandas`
- `os`
- `bitsandbytes`
- `accelerate`
- `peft`
- `transformers`
- `datasets`
- `torch`

### Installation
To install the required dependencies, use the following command:
```bash
pip install numpy pandas bitsandbytes accelerate peft transformers datasets torch
```

### Data Loading and Preprocessing
- The dataset is loaded from a CSV file located in the Kaggle input directory.
- The labels are mapped to integers ('Positive' to 1 and 'Negative' to 0).
- The dataset is tokenized using the `transformers` library.

### Model Configuration
- The model and tokenizer are loaded from the Hugging Face model hub.
- The model is configured for quantization using `BitsAndBytesConfig` to enable efficient training with reduced precision (4-bit).

### Training
- The data is split into training and testing sets.
- A `DataCollatorWithPadding` is used to handle padding.
- The model is trained using the `Trainer` API from the `transformers` library with specified `TrainingArguments`.

### Evaluation
The model is evaluated on the test set to obtain the final performance metrics.

## Results
The project achieved the following scores:

- **Accuracy**: [Insert Accuracy Score]
- **Precision**: [Insert Precision Score]
- **Recall**: [Insert Recall Score]
- **F1 Score**: [Insert F1 Score]

## Usage
To run the notebook and reproduce the results, open `attempt2.ipynb` and execute all cells.

## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for providing the dataset and environment.
- [Hugging Face](https://huggingface.co/) for the `transformers` library and models.
