import argparse
import os
import pickle
import pandas as pd
from model import build_model

#Loading datasets and model

X_TRAIN_NAME = "X_train.zip"
Y_TRAIN_NAME = "y_train.zip"
X_TEST_NAME = "X_test.zip"

DATA_DIR = "data"
PICKLE_NAME = "model.pickle"

#Train function
def train_model():

    X = pd.read_csv(os.sep.join([DATA_DIR, X_TRAIN_NAME]), low_memory=False)
    y = pd.read_csv(os.sep.join([DATA_DIR, Y_TRAIN_NAME]), low_memory=False)

    model = build_model()
    model.fit(X, y.values.ravel())

    # Save to pickle
    with open(PICKLE_NAME, "wb") as f:
        pickle.dump(model, f)

#Test function
def test_model():
    X_test = pd.read_csv(os.sep.join([DATA_DIR, X_TEST_NAME]))

    # Load pickle
    with open(PICKLE_NAME, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    print("### predictions ###")
    print(y_pred)

#main function
def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project."
    )
    parser.add_argument(
        "stage",
        metavar="stage",
        type=str,
        choices=["train", "test"],
        help="Stage to run.",
    )

    stage = parser.parse_args().stage

    if stage == "train":
        print("Training model...")
        train_model()

    elif stage == "test":
        print("Testing model...")
        test_model()


if __name__ == "__main__":
    main()
