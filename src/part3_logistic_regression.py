'''
PART 3: Logistic Regression
'''

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Drop unnecessary columns or columns with datetime data if not needed
    df = df.drop(columns=['arrest_date_univ', 'arrest_date_event'])

    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Preprocess numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocess categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def run_logistic_regression(df_arrests):
    # Ensure the target column 'y' exists
    if 'y' not in df_arrests.columns:
        df_arrests['y'] = df_arrests['charge_degree']  # Adjust this to your target column

    # Preprocess the data
    preprocessor = preprocess_data(df_arrests)
    
    # Split features and target
    X = df_arrests.drop(columns=['y', 'charge_degree'])
    y = df_arrests['y']

    # Check the distribution of the target variable
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"The target variable needs to have at least two classes, but it has only one class: {unique_classes}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the logistic regression model using pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000))])

    param_grid = {
        'classifier__C': [0.1, 1.0, 10, 100],
        'classifier__solver': ['liblinear', 'saga']
    }

    gs_cv = GridSearchCV(clf, param_grid, cv=5)
    gs_cv.fit(X_train, y_train)

    # Make predictions
    y_pred = gs_cv.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")

    # Prepare the result DataFrames
    df_arrests_train = X_train.copy()
    df_arrests_train['y'] = y_train

    df_arrests_test = X_test.copy()
    df_arrests_test['y'] = y_test
    df_arrests_test['pred_lr'] = y_pred

    return df_arrests_train, df_arrests_test, y_pred

# If this script is run standalone, execute the run function
if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(__file__), '../data')
    pred_universe = pd.read_csv(os.path.join(data_directory, 'preprocessed_pred_universe.csv'))
    df_arrests_train, df_arrests_test, pred_lr = run_logistic_regression(pred_universe)
