# SklearnGUI - Machine Learning Model Builder

SklearnGUI is a user-friendly desktop application that allows you to build, train, and evaluate machine learning models without writing code. It provides an intuitive interface for data preprocessing, model selection, and performance evaluation.

---

## Installation

You can install and run SklearnGUI on any system with Python 3.6 or newer. No pre-built executable is needed-just follow these steps:

### 1. Download the Source Code

- Clone the GitHub repository or download and extract the source code ZIP.

### 2. Create a Virtual Environment (Recommended)

A virtual environment keeps your project’s dependencies isolated from other Python projects on your system.

- **Open a terminal or command prompt in the project folder.**
- Run the following command to create a virtual environment named `.venv`:

  ```bash
  python -m venv SklearnGUIENV
  ```

- **Activate the virtual environment:**

  - On **Windows**:
    ```bash
    SklearnGUIENV\Scripts\activate
    ```
  - On **macOS/Linux**:
    ```bash
    source SklearnGUIENV/bin/activate
    ```
    
### 3. Install Required Libraries

- With the virtual environment activated, install all dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  This will install all necessary packages in bulk[2].

### 4. Run the Application

- Start SklearnGUI with:

  ```bash
  python3 SklearnGUI.py
  ```

---

## Features

**Data Handling**
- CSV Import: Load datasets from CSV files
- Data Preview: View the first 100 rows in a table
- Missing Value Handling: Option to drop NA values

**Data Preprocessing**
- Right-Click Context Menu: Access preprocessing options for columns
- Target Selection: Set any column as the prediction target
- Column Encoding: One-hot, label, and target encoding

**Model Building**
- Model Selection: Choose regression or classification algorithms
- Parameter Configuration: Customize model parameters
- Grid Search: Find optimal hyperparameters automatically

**Model Evaluation**
- Performance Metrics: View comprehensive evaluation metrics
- Visualizations: Actual vs. Predicted plots, confusion matrices
- Model Export: Save trained models for later use

---

## Usage

1. **Download and extract the source code**
2. **Set up a virtual environment and install dependencies as described above**
3. **Run the application:**  
   ```bash
   python3 SklearnGUI.py
   ```
4. **Load your data and start building machine learning models**

---

## Requirements

- Python 3.6 or newer
- All required libraries are listed in `requirements.txt` and will be installed automatically
