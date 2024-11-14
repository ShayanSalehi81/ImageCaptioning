# ImageCaptioning

This repository contains the code and resources for an **Image Captioning** project developed as part of a Machine Learning course. The project implements a deep learning pipeline to automatically generate captions for images using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). Specifically, it leverages a pre-trained ResNet50 model for feature extraction and a custom RNN model for generating captions in natural language.

## Project Overview

Image captioning is a fundamental task in computer vision and natural language processing that involves describing the content of an image in words. This project utilizes the **Flickr8k** dataset, which consists of 8,091 images, each annotated with five captions describing the image content. The goal is to train a model that can generate accurate and meaningful descriptions for new, unseen images.

### Key Features
- **Image feature extraction** using a pre-trained ResNet50 model.
- **Sequence generation** using an RNN model with LSTM layers to produce captions based on image features.
- **Evaluation** using a BERT-based similarity metric to compare generated captions with reference captions.

## Directory Structure

- **Captions**: Contains the `captions.txt` file with image IDs and corresponding captions.
- **Code**:
  - `ImageCaptioning.ipynb`: Jupyter Notebook containing code, explanations, and results of the image captioning model.
  - `ImageCaptioning.py`: Python script version of the notebook for running the model outside of a Jupyter environment.
- **.gitignore**: Specifies files and folders to be ignored by git.
- **LICENSE**: The license for this project.
- **README.md**: Project documentation.

## Installation

To run this project, you need Python 3.7 or higher. The required libraries can be installed using:

```bash
pip install tensorflow keras numpy transformers sklearn pillow
```

Additionally, download and unzip the **Flickr8k** dataset and captions file as instructed in the notebook.

## Usage

### Running the Jupyter Notebook

1. Open `ImageCaptioning.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells step-by-step. Follow the comments to understand each part of the code.
3. Modify the notebook to experiment with different configurations, parameters, or model architectures.

This will train the model and evaluate it on the test set.

## Model Architecture

The project employs a **CNN-RNN architecture** for image captioning:

1. **CNN Feature Extraction**: A pre-trained ResNet50 model (without the final classification layer) extracts a high-dimensional feature vector from each image.
2. **Tokenization and Preprocessing**: Captions are cleaned, tokenized, and transformed into sequences for input to the RNN.
3. **RNN Model**: The RNN is implemented with LSTM layers to generate captions word-by-word. Each word prediction is conditioned on both the previous words and the image feature vector.
4. **Training**: The model is trained to minimize categorical cross-entropy loss, with generated sequences compared against true caption sequences.

## Evaluation

The model generates captions for test images, and the similarity between generated and true captions is calculated using a **BERT-based similarity model**. This model compares the semantic similarity between generated captions and reference captions.

### Example Evaluation Output

For each test image, the model outputs:
- The predicted caption
- The true caption
- The similarity score between them

## Example Results

The notebook includes code to display a few test images with their predicted captions and similarity scores.

## Future Improvements

- **Beam Search Decoding**: Implementing beam search could improve the quality of generated captions by considering multiple candidate captions at each step.
- **Attention Mechanism**: Adding an attention layer could allow the model to focus on different parts of the image when generating different words.
- **Fine-tuning BERT for Similarity Scoring**: Fine-tuning BERT specifically on the image-captioning task could yield better similarity metrics.

## License

This project is licensed under the MIT License. See the LICENSE file for details.