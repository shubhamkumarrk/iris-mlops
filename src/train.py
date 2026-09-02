# importing the requried librairies
import joblib
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder


def train_model():
    # laoding the dataset
    df = pd.read_csv('D:\\iris-mlops\\notebook\\IRIS.csv')

    # drop the duplicate values
    df = df.drop_duplicates()

    # Seprating the features and target variable
    x = df.drop('species',axis=1)
    y = df['species']
    #print(y)

    # encoding the categorical values
    encoder = LabelEncoder()
    encode_y = encoder.fit_transform(y)

    # splitting the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(x, encode_y, test_size=0.2, random_state=42)

    # scalling the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    # training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")

    #joblib.dump(model,"iris_model.pkl")

    #print("IRIS Model as iris_model.pkl")


if __name__=="__main__":
    train_model()

