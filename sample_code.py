import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for initial data cleaning operations.
    Removes columns with too many null values and single-value columns.
    """
    def __init__(self, null_threshold=0.60):
        self.null_threshold = null_threshold
        self.columns_to_drop_ = None
    
    def fit(self, X, y=None):
        to_remove = []
        for col in X.columns:
            # Remove columns with only one unique value
            if X[col].nunique() == 1:
                to_remove.append(col)
            # Remove columns with high percentage of null values
            elif (X[col].isnull()).mean() > self.null_threshold:
                to_remove.append(col)
        
        self.columns_to_drop_ = to_remove
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop_, axis=1)

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers in numeric features using IQR method.
    """
    def __init__(self, k=1.5):
        self.k = k
        self.boundaries = {}
    
    def fit(self, X, y=None):
        # Convert numpy array to pandas DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Calculate boundaries for all columns
        for col in range(X.shape[1]):
            Q1 = np.percentile(X.iloc[:, col].dropna(), 25)
            Q3 = np.percentile(X.iloc[:, col].dropna(), 75)
            IQR = Q3 - Q1
            self.boundaries[col] = {
                'lower': Q1 - self.k * IQR,
                'upper': Q3 + self.k * IQR
            }
        return self
    
    def transform(self, X):
        # Convert numpy array to pandas DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        for col in range(X.shape[1]):
            # Replace outliers with boundaries
            X_copy.iloc[:, col] = np.clip(
                X_copy.iloc[:, col],
                self.boundaries[col]['lower'],
                self.boundaries[col]['upper']
            )
        return X_copy.values

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features that exist in the DataFrame and have appropriate dtypes
    """
    def __init__(self):
        self.numeric_features_ = None
        self.categorical_features_ = None
        
    def fit(self, X, y=None):
        # Identify numeric and categorical columns
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return self
        
    def transform(self, X):
        return X
        
    def get_feature_names(self):
        return self.numeric_features_, self.categorical_features_

class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    A Label Encoder that can handle missing values and unknown categories
    """
    def __init__(self):
        self.encoders = {}
        
    def fit(self, X, y=None):
        # Create an encoder for each column
        for column in X.columns:
            le = LabelEncoder()
            unique_values = X[column].fillna('missing').unique()
            le.fit(unique_values)
            self.encoders[column] = le
        return self
        
    def transform(self, X):
        X_encoded = X.copy()
        for column in X.columns:
            X_encoded[column] = X_encoded[column].fillna('missing')
            known_values = set(self.encoders[column].classes_)
            X_encoded[column] = X_encoded[column].map(lambda x: x if x in known_values else 'missing')
            X_encoded[column] = self.encoders[column].transform(X_encoded[column])
        return X_encoded.values

def load_and_explore_data(filepath):
    """
    Loads the dataset and performs initial exploration.
    Returns DataFrame and basic statistics.
    """
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    return df

def visualize_data(df):
    """
    Creates visualizations for data analysis.
    """
    # Plot null values
    plt.figure(figsize=(12, 6))
    df.isnull().sum().plot.bar()
    plt.title('Null Values by Column')
    plt.xticks(rotation=90)
    plt.show()

    # Target distribution
    plt.figure(figsize=(8, 5))
    sb.histplot(df['target'], kde=True)
    plt.title('Target Distribution')
    plt.show()

    # Target boxplot
    plt.figure(figsize=(8, 5))
    sb.boxplot(df['target'])
    plt.title('Target Boxplot')
    plt.show()

    # Correlation heatmap of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    sb.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

def create_pipeline():
    """
    Creates the main ML pipeline including preprocessing and feature engineering steps.
    """
    # Create preprocessing pipelines for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_handler', OutlierHandler()),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', SafeLabelEncoder())
    ])

    # Create main pipeline
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('selector', FeatureSelector()),
        ('preprocessor', ColumnTransformer([
            ('num', numeric_transformer, lambda x: x.select_dtypes(include=[np.number]).columns),
            ('cat', categorical_transformer, lambda x: x.select_dtypes(exclude=[np.number]).columns)
        ], remainder='drop'))
    ])

    return pipeline

def train_models(X_train, X_val, y_train, y_val):
    """
    Trains and evaluates multiple models on the preprocessed data.
    Returns dictionary of trained models and their scores.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.01),
        'Ridge': Ridge(alpha=0.01),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        results[name] = {
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_val, val_pred)),
            'train_r2': metrics.r2_score(y_train, train_pred),
            'val_r2': metrics.r2_score(y_val, val_pred),
            'model': model
        }
        
        print(f"\n{name} Results:")
        print(f"Train RMSE: {results[name]['train_rmse']:.4f}")
        print(f"Val RMSE: {results[name]['val_rmse']:.4f}")
        print(f"Train R2: {results[name]['train_r2']:.4f}")
        print(f"Val R2: {results[name]['val_r2']:.4f}")
    
    return results

def main():
    """
    Main execution function.
    """
    # Load and explore data
    df = load_and_explore_data('Zillow.csv')
    
    # Print available columns after loading
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Visualize initial data
    # visualize_data(df)
    
    # Handle target variable outliers before splitting
    target_handler = OutlierHandler(k=3)  # Using k=3 for more lenient outlier removal
    target_mask = (df['target'] > df['target'].quantile(0.01)) & \
                 (df['target'] < df['target'].quantile(0.99))
    df = df[target_mask]
    
    # Create and fit pipeline
    pipeline = create_pipeline()
    
    # Split features and target
    X = df.drop(['parcelid', 'target'], axis=1)
    y = df['target'].values
    
    # Print columns after dropping
    print("\nColumns after dropping parcelid and target:")
    print(X.columns.tolist())
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit and transform the data
    X_train_transformed = pipeline.fit_transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    
    # Train and evaluate models
    model_results = train_models(X_train_transformed, X_val_transformed, y_train, y_val)
    
    return model_results

if __name__ == "__main__":
    model_results = main()
