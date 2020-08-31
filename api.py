import os
import flask
import pickle
import argparse
import numpy as np
import pandas as pd
from flask import request, jsonify
from statsmodels.discrete.discrete_model import LogitResults

# define root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = flask.Flask(__name__)
app.config['DEBUG'] = True

# load model
model = LogitResults.load(f'{ROOT_DIR}/customer_segmentation_model.pkl')

# load features of interest
with open(f'{ROOT_DIR}/features.pkl', 'rb') as file:
    fInfo = pickle.load(file)


std_scaler = fInfo['scaler']
imputer = fInfo['imputer']
features = fInfo['features']


@app.route('/predict', methods=['POST'])
def predictAd():
    """
    Receives parameters to model and returns
    business outcome :int: - customer receives advertisement
    phat :float: - predicted probability of customer purchasing product
    params: features passed to the model
    """

    #TODO validate request
    #TODO handle batch calls
    req = request.get_json()

    # flag to determine if request is a batch call
    isBatch = False

    # batch call
    if type(req) == list:
        index = list(range(len(req)))
        isBatch = True
    else:
        index = [0]

    df_test = pd.DataFrame(req, index=index)

    # remove unwanted chars
    df_test['x12'] = df_test['x12'].str.replace('$', '')
    df_test['x12'] = df_test['x12'].str.replace(',', '')
    df_test['x12'] = df_test['x12'].str.replace(')', '')
    df_test['x12'] = df_test['x12'].str.replace('(', '-')
    df_test['x12'] = df_test['x12'].astype(float)
    df_test['x63'] = df_test['x63'].str.replace('%', '').astype(float)

    # mean fill NaN values
    df_no_obj = df_test.drop(columns=['x5', 'x31', 'x81', 'x82'])

    df_fill = pd.DataFrame(data=imputer.statistics_.reshape(1, -1), columns=list(df_no_obj.columns))

    test_imputed = df_no_obj.fillna(df_fill)

    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)

    # discretize necessary object columns and concat to df
    dumb5 = pd.get_dummies(df_test['x5'], prefix='x5', prefix_sep='_', dummy_na=True)
    test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df_test['x31'], prefix='x31', prefix_sep='_', dummy_na=True)
    test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df_test['x81'], prefix='x81', prefix_sep='_', dummy_na=True)
    test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    # set missing features to 0
    fin = set(features)
    fih = set(list(test_imputed_std.columns))

    # missing features not in data
    fm = list(fin - fih)

    test_imputed_std[fm] = 0

    test_imputed_std[fm] = test_imputed_std[fm].astype(np.int)

    test_imputed_std = test_imputed_std[features]

    # pass params through model and return result
    df_preds = pd.DataFrame(model.predict(test_imputed_std.values), columns=['phat'])
    df_preds['business_outcome'] = np.zeros(df_preds.shape[0], dtype=int)
    df_preds.loc[df_preds.phat >= .75, 'business_outcome'] = 1
    df_preds['params'] = test_imputed_std.to_dict('records')

    return jsonify(df_preds.to_dict('records'))

@app.route('/', methods=['GET'])
def index():
    return "<h1> I'm here </h1>"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', type=str, required=True, help='ip address of device')
    parser.add_argument('-port', type=int, required=True, help='port number to serve api on')
    args = parser.parse_args()

    # start app
    app.run(host=args.ip, port=args.port)

