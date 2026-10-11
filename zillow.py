import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")

    missing_percentages = df.isnull().mean()
    saved_column = missing_percentages[missing_percentages < 0.6].index
    df = df[saved_column]

    single_col = [col for col in df.columns if df[col].nunique() == 1] #comprehension
    df = df.drop(single_col, axis = 1)

    return df

def visualize_data(df):
    df['target']
    plt.figure(figsize = (12, 6))
    plt.hist(df["target"])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Targets")
    plt.legend()

    numerical_cols = df.select_dtypes(include = [np.number]).columns
    correlations = df[numerical_cols].corr()['target'].sort_values(ascending = False)
    plt.figure(figsize = (12, 6))
    correlations.plot(kind = "bar")
    plt.title("Feature Correlations with Target")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    # plt.show()

def prepare_features(df):
    #remove the parcelid, target from the input value
    X = df.drop(["parcelid", "target"], axis = 1)
    y = df['target']

    #identify numeric and categotical columns
    numerical_cols = X.select_dtypes(include = [np.number]).columns

    #process numeric features
    numeric_data = X[numerical_cols].copy()
    
    imputer = SimpleImputer(strategy = 'median')
    numeric_data = pd.DataFrame(
        imputer.fit_transform(numeric_data),
        columns = numeric_data.columns
    )

    #scale features to have mean = 0 and variance = 1
    scaler = StandardScaler()
    numeric_data = pd.DataFrame(
        scaler.fit_transform(numeric_data),
        columns = numeric_data.columns
    )

    #process categorical features
    #convert categories to numbers (label encoding)
    #fill missing values with 'missing' category first
    categorical_col = X.select_dtypes(exclude = [np.number]).columns
    categorical_data = X[categorical_col].copy()
    for col in categorical_data:
        categorical_data[col] = pd.Categorical(
            categorical_data[col].fillna("missing")
        ).codes

    #concatenate
    X_processed = pd.concat([numeric_data, categorical_data], axis = 1)
    return X_processed, y

def train_eval_models(X_train, X_test, y_train, y_test):
    #Define the models
    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression" : Ridge(alpha = 1.0),
        "Random Forest" : RandomForestRegressor(
            n_estimators = 10,
            random_state = 42, n_jobs = -1),
    }

    #train and eval each model
    results = {}
    for key, value in models.items():
        print(f"Training: {key}")
        value.fit(X_train, y_train)
        y_prediction = value.predict(X_train)
        y_validation = value.predict(X_test)
        mse = metrics.mean_squared_error(y_train, y_prediction)
        mse_2 = metrics.mean_squared_error(y_test, y_validation)
        r2_scores = metrics.r2_score(y_train, y_prediction)
        r2_scores_2 = metrics.r2_score(y_test, y_validation)

        print(f'MSE: {mse:.5f}')
        print(f'MSE_2: {mse_2:.5f}')
        print(f'R2: {r2_scores:.5f}')
        print(f'R2: {r2_scores_2:.5f}')
             
    return results

def main():
    print("Loading the data")
    df = load_data("Zillow.csv")

    print("Visual")
    visualize_data(df)

    print("Preparing features")
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print("Training")
    final = train_eval_models(X_train, X_test, y_train, y_test)
    return final
    
if __name__ == "__main__":
    main()


#create visualization, implement gradient descent
#use xgboost for grad des