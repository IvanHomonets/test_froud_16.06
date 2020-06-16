import pandas as pd
from flask import Flask, jsonify, request
import pickle


import numpy as np
import pandas as pd


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
