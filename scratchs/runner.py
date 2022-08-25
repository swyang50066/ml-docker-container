import os
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def run():
    # Get environment parameters
    model_path = os.environ["MODEL_PATH"] 

    # Load sample dataset
    df = pd.read_csv("./data.csv", index_col=0)

    # Batch train dataset
    X_train, y_train = (
        np.array(list(df["col1"].values)).reshape(-1, 1), 
        np.array(list(df["col2"].values)).reshape(-1, 1)
    )

    # Batch test dataset
    X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    y_test = X_test**2.

    # Transform data to have polynomial features
    feature_func = PolynomialFeatures(degree=2, include_bias=True)
    X_train_poly = feature_func.fit_transform(X_train)

    # Build model
    model = LinearRegression()

    # Train model
    model.fit(X_train_poly, y_train)

    # Test model
    X_test_poly = feature_func.transform(X_test)
    y_pred = model.predict(X_test_poly)

    # Display result
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    axes.scatter(X_train, y_train, color="green")
    axes.scatter(X_test, y_test, color="blue", alpha=0.5)

    fig.tight_layout()

    plt.show()
    plt.clf()

    # Save result
    output = pd.DataFrame(dict
        (
            col1=X_test.ravel(),
            col2=y_test.ravel(),
            col3=y_pred.ravel()
        )
    )
    output.to_csv(os.path.join(os.getcwd(), "result.csv")) 

    # Save model
    joblib.dump(
        model, os.path.join(model_path, "poly_regressor.pkl")
    )



if __name__ == "__main__":
    run()
