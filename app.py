import pandas as pd
from flask import Flask, jsonify, request
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc, classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import json
from flask import Response

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])


def predict():
    # get data
    data = request.get_json(force=True)
    print(data)

    # convert data into dataframe
    data.update((x, y) for x, y in data.items())
    print(data)
    data_df = pd.DataFrame.from_dict(data)
    print(data_df)

    # predictions
    result = model.predict(data_df)
    print(result)

    # send back to browser
    output = {'transaction': int(result[0])}

    # return data

    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
