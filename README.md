# SklearnGUI - Machine Learning Model Builder

SklearnGUI is a user-friendly desktop application that allows you to build, train, and evaluate machine learning models without writing code. It provides an intuitive interface for data preprocessing, model selection, and performance evaluation.

## Features

### Data Handling
- **CSV Import**: Load your datasets directly from CSV files
- **Data Preview**: View the first 100 rows of your dataset in a tabular format
- **Missing Value Handling**: Option to automatically drop NA values

### Data Preprocessing
- **Right-Click Context Menu**: Access powerful preprocessing options by right-clicking on columns
- **Target Selection**: Set any column as your prediction target with a simple right-click
- **Column Encoding**:
  - One-hot encoding for categorical variables
  - Label encoding to convert categories to integers
  - Target encoding for high-cardinality categorical features
- **Column Deletion**: Exclude irrelevant columns from your analysis
- **Data Splitting**: Easily split your data into training and test sets with adjustable test size and random seed

### Feature Engineering
- **Feature Scaling**: Multiple scaling options for both features and target
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
  - Normalizer
- **Target Transformation**: Apply log transformations to handle skewed target distributions

### Model Building
- **Model Selection**: Choose from a variety of algorithms for both regression and classification tasks
- **Parameter Configuration**: Customize model parameters through an intuitive interface
- **Grid Search**: Automatically find optimal hyperparameters using grid search

### Regression Models
- Linear Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Neural Network Regressor (MLP)

### Classification Models
- Logistic Regression
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- Neural Network Classifier (MLP)

### Model Evaluation
- **Performance Metrics**: View comprehensive metrics for your trained model
  - Regression: MSE, RMSE, MAE, R² Score
  - Classification: Accuracy, Precision, Recall, F1 Score
- **Visualizations**: 
  - Regression: Actual vs. Predicted scatter plot
  - Classification: Confusion matrix
- **Model Export**: Save your trained models for later use

## Installation

No installation required! SklearnGUI is distributed as a standalone binary executable that includes all necessary dependencies.

## Usage

1. **Launch the Application**: Double-click the `SklearnGUI` executable file
2. **Load Your Data**: Click "Dataset Upload" and select your CSV file
3. **Preprocess Your Data**:
   - Right-click on column headers to set target or apply encodings
   - Configure preprocessing options in the left panel
4. **Split Your Data**: Click "Split Data" to create training and test sets
5. **Select Model Type**: Choose between regression and classification
6. **Configure and Train**: Select a model, adjust parameters, and click "Train Model"
7. **Evaluate Performance**: Click "See Performance" to view metrics and visualizations
8. **Save Your Model**: Check "Save Model After Training" to export your trained model

## Requirements

None! The executable contains all required libraries and dependencies.

## Notes

- For large datasets, processing may take some time depending on your system specifications
- The application automatically detects non-numeric columns that need encoding
- Right-click functionality is essential for many preprocessing operations
