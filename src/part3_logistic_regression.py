'''
PART 3: Logistic Regression
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_logistic_regression(df_arrests):
    # Ensure the target column 'y' is present
    if 'y' not in df_arrests.columns:
        raise ValueError("The target column 'y' is not present in the dataframe.")
    
    # Split the data
    X = df_arrests.drop(columns=['y'])
    y = df_arrests['y']

    # Check the distribution of the target variable
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"The target variable needs to have at least two classes, but it has only one class: {unique_classes}")

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Define the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Define the model and parameters
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])
    
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100, 1000],
        'classifier__solver': ['liblinear', 'saga']
    }
    
    gs_cv = GridSearchCV(model, param_grid, cv=5)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    pred_universe = pd.read_feather(os.path.join(data_directory, 'preprocessed_pred_universe.feather'))
    df_arrests_train, df_arrests_test, pred_lr = run_logistic_regression(pred_universe)
