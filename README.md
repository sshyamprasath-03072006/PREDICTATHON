# PREDICTATHON

## Project Title:Deepfake Detection Model

## Project Description:
This project aims to build a deepfake detection model that can classify images as either real or fake. The model is based on Convolutional Neural Networks (CNNs), designed to analyze and learn patterns from images in order to predict their authenticity. The dataset consists of real and fake images, and the model outputs predictions based on the features learned from these images. This model is useful for identifying manipulated images, a growing concern in fields such as digital media, security, and law enforcement.

## Features:
- **Image Preprocessing**: Resize, normalize, and split data into training and test sets.
- **Model Architecture**: CNN-based architecture for binary classification (real or fake).
- **Prediction Generation**: Outputs predictions for test images and saves them in a structured JSON file.
- **Output**: JSON file with predicted labels for each image in the test dataset.

## Installation
Install the required dependencies using **pip**:

    
    pip install -r requirements.txt
    

## Running the Code

1. Have the data folder in the same directory
2. Run the code now
3. Prediction json file is now created
