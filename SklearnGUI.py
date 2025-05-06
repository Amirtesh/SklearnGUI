import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QFileDialog, QTableView, QCheckBox, QComboBox, 
                            QDoubleSpinBox, QLineEdit, QScrollArea, QSplitter, QHeaderView,
                            QTableWidget, QTableWidgetItem, QMenu, QAction, QGridLayout, 
                            QGroupBox, QTabWidget, QMessageBox, QProgressBar, QSpinBox)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QFont
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                             AdaBoostRegressor, AdaBoostClassifier,
                             GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DataFrameModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]
    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

class MLModelBuilder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.target_column = None
        self.deleted_columns = []
        self.param_widgets = {}
        self.grid_widgets = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('ML Model Builder')
        self.setGeometry(100, 100, 1200, 800)
        main_layout = QHBoxLayout()
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        self.upload_btn = QPushButton('Dataset Upload')
        self.upload_btn.clicked.connect(self.upload_dataset)
        left_layout.addWidget(self.upload_btn)
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel('Test Size:'))
        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.1, 0.5)
        self.test_size_input.setSingleStep(0.1)
        self.test_size_input.setValue(0.2)
        test_size_layout.addWidget(self.test_size_input)
        test_size_layout.addWidget(QLabel('Random Seed:'))
        self.random_seed_input = QSpinBox()
        self.random_seed_input.setRange(0, 9999)
        self.random_seed_input.setValue(42)
        test_size_layout.addWidget(self.random_seed_input)
        left_layout.addLayout(test_size_layout)
        self.drop_na_cb = QCheckBox('Drop NA Values')
        left_layout.addWidget(self.drop_na_cb)
        self.split_btn = QPushButton('Split Data')
        self.split_btn.clicked.connect(self.split_data)
        self.split_btn.setEnabled(False)
        left_layout.addWidget(self.split_btn)
        scale_x_group = QGroupBox("Feature Scaling (X)")
        scale_x_layout = QVBoxLayout()
        self.scale_x_cb = QCheckBox('Scale X')
        scale_x_layout.addWidget(self.scale_x_cb)
        self.scale_x_method = QComboBox()
        self.scale_x_method.addItems(['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scale_x_layout.addWidget(self.scale_x_method)
        scale_x_group.setLayout(scale_x_layout)
        left_layout.addWidget(scale_x_group)
        scale_y_group = QGroupBox("Target Scaling (y)")
        scale_y_layout = QVBoxLayout()
        self.scale_y_cb = QCheckBox('Scale y')
        scale_y_layout.addWidget(self.scale_y_cb)
        self.scale_y_method = QComboBox()
        self.scale_y_method.addItems(['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'Log1p', 'Log10'])
        scale_y_layout.addWidget(self.scale_y_method)
        scale_y_group.setLayout(scale_y_layout)
        left_layout.addWidget(scale_y_group)
        model_type_group = QGroupBox("Model Type")
        model_type_layout = QVBoxLayout()
        self.regression_cb = QCheckBox('Regression')
        self.classification_cb = QCheckBox('Classification')
        self.regression_cb.toggled.connect(lambda: self.toggle_model_type(self.regression_cb, self.classification_cb))
        self.classification_cb.toggled.connect(lambda: self.toggle_model_type(self.classification_cb, self.regression_cb))
        model_type_layout.addWidget(self.regression_cb)
        model_type_layout.addWidget(self.classification_cb)
        model_type_group.setLayout(model_type_layout)
        left_layout.addWidget(model_type_group)
        self.model_selection = QComboBox()
        self.model_selection.setEnabled(False)
        left_layout.addWidget(QLabel('Select Model:'))
        left_layout.addWidget(self.model_selection)
        params_group = QGroupBox("Model Parameters")
        self.params_layout = QGridLayout()
        params_group.setLayout(self.params_layout)
        left_layout.addWidget(params_group)
        self.grid_search_cb = QCheckBox('Use Grid Search')
        self.grid_search_cb.toggled.connect(self.update_parameters)
        left_layout.addWidget(self.grid_search_cb)
        self.train_btn = QPushButton('Train Model')
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        left_layout.addWidget(self.train_btn)
        self.save_model_cb = QCheckBox('Save Model After Training')
        left_layout.addWidget(self.save_model_cb)
        self.perf_btn = QPushButton('See Performance')
        self.perf_btn.clicked.connect(self.show_performance)
        self.perf_btn.setEnabled(False)
        left_layout.addWidget(self.perf_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        self.tab_widget = QTabWidget()
        self.data_view_widget = QWidget()
        data_view_layout = QVBoxLayout(self.data_view_widget)
        self.data_info_label = QLabel('No dataset loaded')
        data_view_layout.addWidget(self.data_info_label)
        self.table_scroll = QScrollArea()
        self.table_scroll.setWidgetResizable(True)
        self.table_container = QWidget()
        self.table_layout = QVBoxLayout(self.table_container)
        self.data_table = QTableWidget()
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self.show_context_menu)
        self.table_layout.addWidget(self.data_table)
        self.table_scroll.setWidget(self.table_container)
        data_view_layout.addWidget(self.table_scroll)
        self.tab_widget.addTab(self.data_view_widget, "Data View")
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_table = QTableWidget()
        self.results_layout.addWidget(self.results_table)
        self.viz_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.results_layout.addWidget(self.viz_canvas)
        self.tab_widget.addTab(self.results_widget, "Results")
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.tab_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setup_model_selections()
        self.show()

    def setup_model_selections(self):
        self.regression_models = {
            'Linear Regression': LinearRegression,
            'KNN Regressor': KNeighborsRegressor,
            'Decision Tree Regressor': DecisionTreeRegressor,
            'Random Forest Regressor': RandomForestRegressor,
            'AdaBoost Regressor': AdaBoostRegressor,
            'Gradient Boosting Regressor': GradientBoostingRegressor,
            'Neural Network Regressor': MLPRegressor
        }
        self.classification_models = {
            'Logistic Regression': LogisticRegression,
            'KNN Classifier': KNeighborsClassifier,
            'Decision Tree Classifier': DecisionTreeClassifier,
            'Random Forest Classifier': RandomForestClassifier,
            'AdaBoost Classifier': AdaBoostClassifier,
            'Gradient Boosting Classifier': GradientBoostingClassifier,
            'Neural Network Classifier': MLPClassifier
        }
        self.model_params = {
            'Linear Regression': {
                'fit_intercept': ('checkbox', True),
                'n_jobs': ('spinner', -1, -1, 16)
            },
            'Logistic Regression': {
                'C': ('spinbox', 1.0, 0.001, 1000, 0.001),
                'penalty': ('combobox', ['l2', 'l1', 'elasticnet', 'none']),
                'solver': ('combobox', ['lbfgs', 'liblinear', 'saga']),
                'max_iter': ('spinner', 100, 10, 10000)
            },
            'KNN Regressor': {
                'n_neighbors': ('spinner', 5, 1, 50),
                'weights': ('combobox', ['uniform', 'distance']),
                'algorithm': ('combobox', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': ('spinner', 2, 1, 5)
            },
            'KNN Classifier': {
                'n_neighbors': ('spinner', 5, 1, 50),
                'weights': ('combobox', ['uniform', 'distance']),
                'algorithm': ('combobox', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': ('spinner', 2, 1, 5)
            },
            'Decision Tree Regressor': {
                'criterion': ('combobox', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                'max_depth': ('spinner', 10, 1, 100),
                'min_samples_split': ('spinner', 2, 1, 20),
                'min_samples_leaf': ('spinner', 1, 1, 20)
            },
            'Decision Tree Classifier': {
                'criterion': ('combobox', ['gini', 'entropy']),
                'max_depth': ('spinner', 10, 1, 100),
                'min_samples_split': ('spinner', 2, 1, 20),
                'min_samples_leaf': ('spinner', 1, 1, 20)
            },
            'Random Forest Regressor': {
                'n_estimators': ('spinner', 100, 10, 1000),
                'criterion': ('combobox', ['squared_error', 'absolute_error']),
                'max_depth': ('spinner', 10, 1, 100),
                'min_samples_split': ('spinner', 2, 1, 20),
                'n_jobs': ('spinner', -1, -1, 16)
            },
            'Random Forest Classifier': {
                'n_estimators': ('spinner', 100, 10, 1000),
                'criterion': ('combobox', ['gini', 'entropy']),
                'max_depth': ('spinner', 10, 1, 100),
                'min_samples_split': ('spinner', 2, 1, 20),
                'n_jobs': ('spinner', -1, -1, 16)
            },
            'AdaBoost Regressor': {
                'n_estimators': ('spinner', 50, 10, 500),
                'learning_rate': ('spinbox', 1.0, 0.01, 5.0, 0.01),
                'loss': ('combobox', ['linear', 'square', 'exponential'])
            },
            'AdaBoost Classifier': {
                'n_estimators': ('spinner', 50, 10, 500),
                'learning_rate': ('spinbox', 1.0, 0.01, 5.0, 0.01),
                'algorithm': ('combobox', ['SAMME', 'SAMME.R'])
            },
            'Gradient Boosting Regressor': {
                'n_estimators': ('spinner', 100, 10, 1000),
                'learning_rate': ('spinbox', 0.1, 0.01, 1.0, 0.01),
                'max_depth': ('spinner', 3, 1, 32),
                'subsample': ('spinbox', 1.0, 0.1, 1.0, 0.1)
            },
            'Gradient Boosting Classifier': {
                'n_estimators': ('spinner', 100, 10, 1000),
                'learning_rate': ('spinbox', 0.1, 0.01, 1.0, 0.01),
                'max_depth': ('spinner', 3, 1, 32),
                'subsample': ('spinbox', 1.0, 0.1, 1.0, 0.1)
            },
            'Neural Network Regressor': {
                'hidden_layer_sizes': ('text', '100,'),
                'activation': ('combobox', ['relu', 'identity', 'logistic', 'tanh']),
                'solver': ('combobox', ['adam', 'sgd', 'lbfgs']),
                'alpha': ('spinbox', 0.0001, 0.00001, 1.0, 0.00001),
                'learning_rate': ('combobox', ['constant', 'invscaling', 'adaptive']),
                'max_iter': ('spinner', 200, 10, 2000)
            },
            'Neural Network Classifier': {
                'hidden_layer_sizes': ('text', '100,'),
                'activation': ('combobox', ['relu', 'identity', 'logistic', 'tanh']),
                'solver': ('combobox', ['adam', 'sgd', 'lbfgs']),
                'alpha': ('spinbox', 0.0001, 0.00001, 1.0, 0.00001),
                'learning_rate': ('combobox', ['constant', 'invscaling', 'adaptive']),
                'max_iter': ('spinner', 200, 10, 2000)
            }
        }
        self.model_selection.currentTextChanged.connect(self.update_parameters)

    def toggle_model_type(self, checkbox, other_checkbox):
        if checkbox.isChecked():
            other_checkbox.setChecked(False)
            self.model_selection.clear()
            if checkbox == self.regression_cb:
                self.model_selection.addItems(self.regression_models.keys())
            else:
                self.model_selection.addItems(self.classification_models.keys())
            self.model_selection.setEnabled(True)
            self.update_parameters(self.model_selection.currentText())
            self.train_btn.setEnabled(True if self.X_train is not None else False)
        else:
            self.model_selection.clear()
            self.model_selection.setEnabled(False)
            self.clear_parameters()
            self.train_btn.setEnabled(False)

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.display_data()
                self.split_btn.setEnabled(True)
                rows, cols = self.df.shape
                self.data_info_label.setText(f"Loaded dataset: {rows} rows, {cols} columns")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def display_data(self):
        if self.df is None:
            return
        self.data_table.clear()
        rows, cols = self.df.shape
        self.data_table.setRowCount(min(rows, 100))
        self.data_table.setColumnCount(cols)
        self.data_table.setHorizontalHeaderLabels(self.df.columns)
        for i in range(min(rows, 100)):
            for j in range(cols):
                self.data_table.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))
        self.data_table.resizeColumnsToContents()

    def show_context_menu(self, position):
        if self.df is None:
            return
        menu = QMenu()
        col_idx = self.data_table.horizontalHeader().logicalIndexAt(position.x())
        if 0 <= col_idx < len(self.df.columns):
            col_name = self.df.columns[col_idx]
            
            set_target = QAction(f"Set '{col_name}' as target (y)", self)
            set_target.triggered.connect(lambda: self.set_target_column(col_name))
            delete_col = QAction(f"Delete '{col_name}'", self)
            delete_col.triggered.connect(lambda: self.delete_column(col_name))
            
            encoding_menu = QMenu("Encode Column", self)
            
            one_hot_action = QAction("One-hot encode", self)
            one_hot_action.triggered.connect(lambda: self.encode_column(col_name, 'one-hot'))
            encoding_menu.addAction(one_hot_action)
            
            label_action = QAction("Label encode", self)
            label_action.triggered.connect(lambda: self.encode_column(col_name, 'label'))
            encoding_menu.addAction(label_action)
            
            target_action = QAction("Target encode", self)
            target_action.triggered.connect(lambda: self.encode_column(col_name, 'target'))
            encoding_menu.addAction(target_action)
            
            menu.addMenu(encoding_menu)
            menu.addAction(set_target)
            menu.addAction(delete_col)
            menu.exec_(self.data_table.viewport().mapToGlobal(position))

    def encode_column(self, col_name, encoding_type):
        if self.df is None or col_name not in self.df.columns:
            QMessageBox.warning(self, "Error", f"Column '{col_name}' not found in the dataset")
            return
        
        try:
            if encoding_type == 'one-hot':
                one_hot = pd.get_dummies(self.df[col_name], prefix=col_name, dtype=int)
                self.df = pd.concat([self.df, one_hot], axis=1)
                self.df = self.df.drop(col_name, axis=1)
                QMessageBox.information(self, "Encoding Complete", 
                                       f"Applied one-hot encoding to '{col_name}'. Created {one_hot.shape[1]} new columns.")
            
            elif encoding_type == 'label':
                encoder = LabelEncoder()
                self.df[col_name] = encoder.fit_transform(self.df[col_name])
                QMessageBox.information(self, "Encoding Complete", 
                                       f"Applied label encoding to '{col_name}'. Values mapped to integers 0-{len(encoder.classes_)-1}.")
            
            elif encoding_type == 'target':
                if self.target_column is None:
                    QMessageBox.warning(self, "Error", "Please set a target column first for target encoding")
                    return
                    
                if self.df[self.target_column].dtype in ['object', 'category']:
                    target_means = self.df.groupby(col_name)[self.target_column].value_counts(normalize=True).unstack()
                    if target_means.shape[1] == 2:
                        mapping = target_means.iloc[:, 1].to_dict()
                    else:
                        mapping = target_means.max(axis=1).to_dict()
                else:
                    mapping = self.df.groupby(col_name)[self.target_column].mean().to_dict()
                
                self.df[col_name] = self.df[col_name].map(mapping)
                
                if self.df[col_name].isna().any():
                    overall_mean = self.df[self.target_column].mean() if self.df[self.target_column].dtype not in ['object', 'category'] else 0.5
                    self.df[col_name].fillna(overall_mean, inplace=True)
                
                QMessageBox.information(self, "Encoding Complete", f"Applied target encoding to '{col_name}'.")
            
            self.display_data()
            
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.train_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to encode column: {str(e)}")

    def set_target_column(self, col_name):
        self.target_column = col_name
        for col in range(self.data_table.columnCount()):
            if self.data_table.horizontalHeaderItem(col).text() == col_name:
                for row in range(self.data_table.rowCount()):
                    item = self.data_table.item(row, col)
                    if item:
                        item.setBackground(Qt.green)
                break

    def delete_column(self, col_name):
        if col_name not in self.deleted_columns:
            self.deleted_columns.append(col_name)
            for col in range(self.data_table.columnCount()):
                if self.data_table.horizontalHeaderItem(col).text() == col_name:
                    self.data_table.hideColumn(col)
                    break
            QMessageBox.information(self, "Column Deleted", f"'{col_name}' will be excluded from the model")

    def convert_to_numeric(self, df):
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        return df

    def split_data(self):
        if self.df is None or self.target_column is None:
            QMessageBox.warning(self, "Error", "Please load a dataset and set a target column")
            return
        try:
            df_filtered = self.df.drop(columns=self.deleted_columns)
            
            if self.drop_na_cb.isChecked():
                df_filtered = df_filtered.dropna()
            
            non_numeric_cols = []
            for col in df_filtered.columns:
                if col != self.target_column:
                    try:
                        pd.to_numeric(df_filtered[col])
                    except:
                        non_numeric_cols.append(col)
            
            if non_numeric_cols:
                msg = f"Found non-numeric columns that need encoding:\n{', '.join(non_numeric_cols)}\n\nPlease encode these columns before splitting."
                QMessageBox.warning(self, "Non-numeric Columns Detected", msg)
                return
            
            X = df_filtered.drop(columns=[self.target_column])
            X = self.convert_to_numeric(X)
            y = df_filtered[self.target_column]
            
            test_size = self.test_size_input.value()
            random_seed = self.random_seed_input.value()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
            QMessageBox.information(self, "Data Split", f"Data split successfully:\nX_train: {self.X_train.shape}\nX_test: {self.X_test.shape}\ny_train: {self.y_train.shape}\ny_test: {self.y_test.shape}")
            
            if self.regression_cb.isChecked() or self.classification_cb.isChecked():
                self.train_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to split data: {str(e)}")

    def update_parameters(self, model_name):
        self.clear_parameters()
        if not model_name:
            return
        params = self.model_params.get(model_name, {})
        row = 0
        for param_name, param_config in params.items():
            param_type = param_config[0]
            self.params_layout.addWidget(QLabel(param_name), row, 0)
            if param_type == 'checkbox':
                widget = QCheckBox()
                widget.setChecked(param_config[1])
                self.params_layout.addWidget(widget, row, 1)
            elif param_type == 'spinner':
                widget = QSpinBox()
                widget.setValue(param_config[1])
                widget.setRange(param_config[2], param_config[3])
                self.params_layout.addWidget(widget, row, 1)
            elif param_type == 'spinbox':
                widget = QDoubleSpinBox()
                widget.setValue(param_config[1])
                widget.setRange(param_config[2], param_config[3])
                widget.setSingleStep(param_config[4])
                self.params_layout.addWidget(widget, row, 1)
            elif param_type == 'combobox':
                widget = QComboBox()
                widget.addItems(param_config[1])
                self.params_layout.addWidget(widget, row, 1)
            elif param_type == 'text':
                widget = QLineEdit(param_config[1])
                self.params_layout.addWidget(widget, row, 1)
            self.param_widgets[param_name] = widget
            if self.grid_search_cb.isChecked() and param_type in ['spinner', 'spinbox']:
                grid_input = QLineEdit()
                grid_input.setPlaceholderText("[values]")
                self.params_layout.addWidget(grid_input, row, 2)
                self.grid_widgets[param_name] = grid_input
            row += 1

    def clear_parameters(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.param_widgets.clear()
        self.grid_widgets.clear()

    def get_param_values(self):
        params = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                try:
                    text = widget.text()
                    if ',' in text:
                        params[param_name] = eval(f"({text})")
                    else:
                        params[param_name] = int(text) if text.isdigit() else text
                except:
                    params[param_name] = widget.text()
        if self.grid_search_cb.isChecked():
            grid_params = {}
            for param_name, widget in self.grid_widgets.items():
                value = widget.text()
                if value:
                    try:
                        parsed = eval(value)
                        grid_params[param_name] = parsed if isinstance(parsed, (list, tuple)) else [parsed]
                    except:
                        grid_params[param_name] = [value]
            return grid_params
        return params

    def get_scaler(self, scaler_name):
        if scaler_name == 'StandardScaler':
            return StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            return MinMaxScaler()
        elif scaler_name == 'RobustScaler':
            return RobustScaler()
        elif scaler_name == 'Normalizer':
            return Normalizer()
        return None

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Error", "Please split your data first")
            return
        
        try:
            X_train = self.X_train.copy()
            X_test = self.X_test.copy()
            
            try:
                X_train = X_train.astype(float)
                X_test = X_test.astype(float)
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Failed to convert data to numeric format: {str(e)}")
                return
            
            if self.scale_x_cb.isChecked():
                scaler = self.get_scaler(self.scale_x_method.currentText())
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            y_train = self.y_train.copy()
            y_test = self.y_test.copy()
            
            if self.scale_y_cb.isChecked():
                method = self.scale_y_method.currentText()
                if method == 'Log1p':
                    y_train = np.log1p(y_train)
                    y_test = np.log1p(y_test)
                    self.y_transformer = 'Log1p'
                elif method == 'Log10':
                    y_train = np.log10(y_train + 1e-10)
                    y_test = np.log10(y_test + 1e-10)
                    self.y_transformer = 'Log10'
                elif method in ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer']:
                    scaler = self.get_scaler(method)
                    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    y_test = scaler.transform(y_test.values.reshape(-1, 1)).flatten()
                    self.y_transformer = scaler
            
            model_name = self.model_selection.currentText()
            params = self.get_param_values()
            
            if self.regression_cb.isChecked():
                model_class = self.regression_models[model_name]
            else:
                model_class = self.classification_models[model_name]
            
            self.progress_bar.setValue(10)
            
            if self.grid_search_cb.isChecked():
                base_model = model_class()
                n_jobs = -1
                if 'n_jobs' in params:
                    n_jobs = params['n_jobs']
                grid_search = GridSearchCV(base_model, params, cv=5, n_jobs=n_jobs, verbose=0)
                self.progress_bar.setValue(20)
                grid_search.fit(X_train, y_train)
                self.progress_bar.setValue(80)
                self.model = grid_search.best_estimator_
                QMessageBox.information(self, "Grid Search Complete", f"Best parameters: {grid_search.best_params_}\nBest score: {grid_search.best_score_:.4f}")
            else:
                self.model = model_class(**params)
                self.progress_bar.setValue(25)
                self.model.fit(X_train, y_train)
                self.progress_bar.setValue(90)
            
            self.X_train_processed = X_train
            self.X_test_processed = X_test
            self.y_train_processed = y_train
            self.y_test_processed = y_test
            self.progress_bar.setValue(100)
            self.perf_btn.setEnabled(True)
            
            if self.save_model_cb.isChecked():
                self.save_model()
            
            QMessageBox.information(self, "Success", "Model training completed successfully!")
        
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Error", f"Failed to train model: {str(e)}")

    def save_model(self):
        try:
            model_name = self.model_selection.currentText().replace(" ", "_")
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", f"{model_name}_model.pkl", "Pickle Files (*.pkl)")
            if file_path:
                joblib.dump(self.model, file_path)
                summary_path = file_path.replace('.pkl', '_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write(f"Model: {self.model_selection.currentText()}\n")
                    f.write(f"Parameters: {self.model.get_params()}\n")
                    if hasattr(self.model, 'feature_importances_'):
                        f.write("\nFeature Importances:\n")
                        for feature, importance in zip(self.X_train.columns, self.model.feature_importances_):
                            f.write(f"{feature}: {importance:.4f}\n")
                QMessageBox.information(self, "Model Saved", f"Model saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def show_performance(self):
        if self.model is None or self.X_test_processed is None:
            QMessageBox.warning(self, "Error", "No trained model available")
            return
        try:
            y_pred = self.model.predict(self.X_test_processed)
            if self.scale_y_cb.isChecked():
                method = self.scale_y_method.currentText()
                if method == 'Log1p':
                    y_pred = np.expm1(y_pred)
                    y_test = np.expm1(self.y_test_processed)
                elif method == 'Log10':
                    y_pred = np.power(10, y_pred) - 1e-10
                    y_test = np.power(10, self.y_test_processed) - 1e-10
                else:
                    y_test = self.y_test_processed
            else:
                y_test = self.y_test_processed
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(['Metric', 'Value'])
            if self.regression_cb.isChecked():
                metrics = {
                    'Mean Squared Error': mean_squared_error(y_test, y_pred),
                    'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
                    'R² Score': r2_score(y_test, y_pred)
                }
                fig = self.viz_canvas.figure
                fig.clear()
                ax = fig.add_subplot(111)
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                fig.suptitle('Actual vs Predicted')
                self.viz_canvas.draw()
            else:
                average = 'binary' if len(np.unique(self.y_test)) == 2 else 'weighted'
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average=average, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average=average, zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, average=average, zero_division=0)
                }
                cm = confusion_matrix(y_test, y_pred)
                fig = self.viz_canvas.figure
                fig.clear()
                ax = fig.add_subplot(111)
                cax = ax.matshow(cm, cmap='Blues')
                fig.colorbar(cax)
                classes = np.unique(np.concatenate([y_test, y_pred]))
                tick_marks = np.arange(len(classes))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                fig.tight_layout()
                self.viz_canvas.draw()
            self.results_table.setRowCount(len(metrics))
            for i, (metric, value) in enumerate(metrics.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(metric))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            self.results_table.resizeColumnsToContents()
            self.tab_widget.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute metrics: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MLModelBuilder()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

