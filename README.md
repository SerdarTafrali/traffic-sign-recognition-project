# Traffic Sign Recognition Project 🚦

This project focuses on developing a machine learning model for traffic sign recognition. Using advanced techniques and a structured pipeline, we aim to accurately classify traffic signs from a given dataset. The project leverages radial basis function (RBF) kernels with support vector machines (SVMs), hyperparameter optimization, and thorough evaluation to achieve high classification performance.

## Key Features of the Project

- **End-to-End Machine Learning Pipeline**:
  - Data preprocessing (resizing, normalization, and splitting).
  - Model training using RBF kernel SVMs.
  - Hyperparameter optimization with `GridSearchCV`.
  - Performance evaluation with metrics like accuracy, precision, recall, and F1-score.
  - Visualization of results using confusion matrices and classification reports.

## Dataset

- The dataset used in this project contains images of various traffic signs. Each image is labeled with its corresponding class.
- Images were resized to 32x32 pixels and normalized for consistency and optimal training.

## Model

- **Base Model**:
  - Initial training was done using default parameters of an SVM with an RBF kernel.
- **Optimized Model**:
  - Hyperparameters such as `C` (regularization) and `gamma` (kernel coefficient) were fine-tuned using `GridSearchCV` to improve performance.

## Evaluation

- The optimized model achieved an accuracy of **91.76%**, with significant improvements in class-specific metrics compared to the base model.
- Detailed analysis of confusion matrices revealed improved classification for both high- and low-support classes.

## Project Workflow

### Data Preprocessing
- Images were resized, normalized, and split into training and test sets.
- The data was further standardized using `StandardScaler` for optimal performance with SVM.

### Base Model Training
- The initial SVM model was trained with default parameters as a benchmark.

### Hyperparameter Optimization
- Using `GridSearchCV`, the parameters `C` and `gamma` were fine-tuned to achieve a balance between bias and variance.

### Performance Evaluation
- Metrics such as accuracy, precision, recall, and F1-score were computed.
- Confusion matrices were visualized to identify misclassifications and model strengths.

### Visualization
- Confusion matrices and classification reports were generated to provide insights into the model’s performance.

## Results

- **Base Model**:
  - Accuracy: **88.22%**
  - Precision: **89.76%**
  - Recall: **88.22%**
  - F1-Score: **88.43%**
- **Optimized Model**:
  - Accuracy: **91.76%**
  - Precision: **92.28%**
  - Recall: **91.76%**
  - F1-Score: **91.78%**

The optimized model demonstrates significant improvements across all metrics, especially in challenging low-support classes.

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - `scikit-learn` for model training and evaluation.
  - `torchvision` for dataset handling and transformations.
  - `matplotlib` and `seaborn` for data visualization.
- **Optimization Technique**: Grid Search for hyperparameter tuning.

## Future Work

- **Data Augmentation**:
  - Apply techniques like rotation, flipping, and cropping to expand the dataset and improve model generalization.
- **Deep Learning Models**:
  - Experiment with Convolutional Neural Networks (CNNs) for potentially higher performance.
- **Deployment**:
  - Package the trained model into a web application or mobile app for real-time traffic sign recognition.
 
## Conclusion
This project showcases a robust machine learning pipeline for traffic sign recognition. It demonstrates the importance of preprocessing, hyperparameter optimization, and thorough evaluation in achieving high classification performance. By expanding on the current results with advanced techniques like deep learning, this project has the potential to be implemented in real-world autonomous driving systems.
