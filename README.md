# MRI Classification Model

This project implements a Convolutional Neural Network (CNN) to classify MRI scans into four categories of cognitive impairment: Mild Impairment, Moderate Impairment, No Impairment, and Very Mild Impairment.

## Dataset
The dataset consists of grayscale MRI images organized into four classes:
- **Mild_Impairment**
- **Moderate_Impairment**
- **No_Impairment**
- **Very_Mild_Impairment**

The dataset should be placed in the following directory structure:
```
D:\downloads\archive\Combined_Dataset\train
    ├── Mild_Impairment
    ├── Moderate_Impairment
    ├── No_Impairment
    ├── Very_Mild_Impairment
```

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pillow tensorflow scikit-learn matplotlib seaborn
```

## Model Architecture
The implemented CNN consists of:
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Flatten and Dense layers for classification

## Training Process
1. Load and preprocess images (grayscale conversion, resizing, and normalization)
2. Convert labels to one-hot encoding
3. Split data into training and validation sets
4. Train the CNN using categorical cross-entropy loss and Adam optimizer
5. Save the trained model as `mri_classification_model.h5`

## Evaluation Metrics
After training, the model is evaluated using:
- **Accuracy**
- **Matthews Correlation Coefficient (MCC)**
- **Confusion Matrix**
- **Precision, Recall, and F1-score**
- **ROC-AUC Score**
- **ROC Curves for each class**

## Running the Model
Execute the script to train the model:
```bash
python train_model.py
```

## Model Performance Visualization
The script generates visualizations including:
- Confusion Matrix
- ROC-AUC Curves for multi-class classification

## Output
After training, the model is saved as:
```
mri_classification_model.h5
```

## Notes
- Ensure the dataset is correctly placed before running the script.
- Modify `target_size` if different image resolutions are used.
- Adjust hyperparameters (epochs, batch size) for better performance.

## License
This project is for research and educational purposes. Modify and use it as needed!

