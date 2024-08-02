'''
PART 3: Logistic Regression
'''

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def run_logistic_regression(df_arrests):
    # Split the data
    X = df_arrests.drop(columns=['y'])
    y = df_arrests['y']

    # Check the distribution of the target variable
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"The target variable needs to have at least two classes, but it has only one class: {unique_classes}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and parameters
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'solver': ['liblinear', 'saga']}
    gs_cv = GridSearchCV(model, param_grid, cv=5)

    # Fit the model
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
