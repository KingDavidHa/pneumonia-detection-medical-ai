# Pneumonia Detection using Deep Learning

![Pneumonia Detection](https://img.shields.io/badge/Medical%20AI-Pneumonia%20Detection-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.x-green)

## Project Overview

This project implements a convolutional neural network (CNN) to detect pneumonia from chest X-ray images. The model is trained to classify chest X-rays into two categories: **NORMAL** and **PNEUMONIA**.

### Key Results
- **Test Accuracy:** 82.53%
- **Pneumonia Precision:** 95% (high clinical relevance)
- **Effective for clinical decision support**

## Dataset Structure

The dataset contains chest X-ray images organized as follows:
```
chest_xray/
├── train/
│   ├── NORMAL/ (1,341 images)
│   └── PNEUMONIA/ (3,885 images)
├── val/
│   ├── NORMAL/ (8 images)
│   └── PNEUMONIA/ (8 images)
└── test/
    ├── NORMAL/ (234 images)
    └── PNEUMONIA/ (390 images)
```

## Model Architecture

The implemented CNN model includes:

- **Three convolutional blocks with:**
  - Conv2D layers with ReLU activation
  - Batch normalization
  - MaxPooling
  - Dropout for regularization
- **Fully connected layers with:**
  - Dense layer with 512 units
  - Final sigmoid output for binary classification

```python
def build_cnn_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

## Data Preprocessing & Augmentation

To improve model robustness, the following preprocessing steps were applied:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

## Model Performance

### Classification Report
```
              precision    recall  f1-score   support
      NORMAL       0.70      0.93      0.80       234
   PNEUMONIA       0.95      0.76      0.84       390
    accuracy                           0.83       624
   macro avg       0.82      0.85      0.82       624
weighted avg       0.86      0.83      0.83       624
```

### Visualizations
The repository includes:
- Confusion matrix
- ROC curve (AUC = 0.89)
- Training and validation accuracy/loss plots

## Strategic Insights & Future Work

### Model Improvements
- Implement transfer learning with pre-trained models (VGG16, ResNet50)
- Address class imbalance with weighted loss functions
- Use k-fold cross-validation for robust evaluation

### Clinical Application Enhancements
- Add explainability with techniques like Grad-CAM
- Design intuitive interfaces for clinical staff
- Develop APIs for hospital system integration

### Business & Deployment Strategy
- Conduct clinical validation studies
- Perform ROI analysis
- Implement phased deployment starting with MVP

### Ethics & Compliance
- Work toward medical device certification
- Ensure HIPAA compliance
- Monitor for algorithmic bias

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- numpy
- pandas

### Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/pneumonia-detection.git

# Change directory
cd pneumonia-detection

# Install required packages
pip install -r requirements.txt
```

### Running the Model
```python
# Import libraries and load data
from google.colab import drive
drive.mount('/content/drive')
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Setting up dataset paths
dataset_path = '/content/drive/MyDrive/chest_xray/chest_xray'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build and train the model
model = build_cnn_model()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## Conclusion

This project demonstrates the potential of deep learning for medical image analysis. The high precision in pneumonia detection (95%) shows promising potential for clinical decision support. The CNN architecture provides a solid foundation that can be extended with transfer learning approaches for improved performance.

### Strategic Analysis & Recommendations

#### Project Performance Summary:
- Test accuracy reached 82.53% with a pneumonia precision of 95%, demonstrating a reliable baseline CNN model.
- From a clinical perspective, the high precision in pneumonia detection indicates strong potential for AI-assisted diagnosis.
- The developed pipeline is scalable and applicable to other medical imaging tasks.

#### Strategic Implications:
1. Medical AI should focus on augmenting clinical decision-making rather than full automation.
2. Radiologists' time can be optimized, leading to greater operational efficiency in hospitals.
3. Establishes a foundational AI infrastructure that is adaptable to other diagnostic domains.

#### Actionable Recommendations:
1. **Model Improvement:**
   - Integrate transfer learning using advanced architectures (e.g., VGG16, ResNet50).
   - Address class imbalance through weighted loss functions and enhanced data augmentation.
   - Employ cross-validation techniques such as k-fold for robust evaluation.

2. **Practical Enhancements:**
   - Implement interpretability techniques like Grad-CAM to increase trust and transparency.
   - Develop intuitive interfaces that are accessible to clinical staff.
   - Design APIs to ensure seamless integration with existing hospital information systems (HIS).

3. **Business & Clinical Strategy:**
   - Conduct pilot studies in real-world clinical settings to validate the AI system's efficacy.
   - Perform ROI analysis to demonstrate cost-effectiveness of deployment.
   - Follow a phased implementation strategy, beginning with an MVP rollout.

4. **Ethics & Regulatory Readiness:**
   - Formulate a strategy to achieve medical device certification and regulatory compliance.
   - Establish strong data privacy protection mechanisms aligned with HIPAA or local regulations.
   - Implement monitoring systems for algorithmic bias and fairness.

This project goes beyond a technical implementation—demonstrating real potential to generate tangible value in clinical environments. To ensure successful deployment of medical AI, excellence in technology must be matched by seamless integration into clinical workflows, clinician trust, and measurable impact on patient care. Going forward, strong collaboration among developers, medical professionals, and strategic decision-makers will be critical to advancing the future of AI in healthcare.

