import os
import flask
import pickle
import argparse
import numpy as np
import pandas as pd
from flask import request, jsonify
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import LogitResults

# define root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = flask.Flask(__name__)
app.config['DEBUG'] = True

# load model
model = LogitResults.load(f'{ROOT_DIR}/customer_segmentation_model.pkl')

# load features of interest
with open(f'{ROOT_DIR}/features.pkl', 'rb') as file:
    features = pickle.load(file)

# load feature means
with open(f'{ROOT_DIR}/feature_means.pkl', 'rb') as file:
    feature_means = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predictAd():
    """
    Receives parameters to model and returns
    business outcome :int: - customer receives advertisement
    phat :float: - predicted probability of customer purchasing product
    model input parameters
    """

    #TODO validate request
    #TODO handle batch calls
    req = request.get_json()
    df_test = pd.DataFrame(req, index=[0])

    # remove unwanted chars
    df_test['x12'] = df_test['x12'].str.replace('$', '')
    df_test['x12'] = df_test['x12'].str.replace(',', '')
    df_test['x12'] = df_test['x12'].str.replace(')', '')
    df_test['x12'] = df_test['x12'].str.replace('(', '-')
    df_test['x12'] = df_test['x12'].astype(float)
    df_test['x63'] = df_test['x63'].str.replace('%', '').astype(float)

    # df_test = df_test[features]

    # mean fill NaN values
    df_no_obj = df_test.drop(columns=['x5', 'x31', 'x81', 'x82'])

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
    # TODO fix edge case where one row has a nan
    test_imputed = pd.DataFrame(imputer.fit_transform(df_no_obj), columns=df_no_obj.columns)

    # center and scale data
    # std_scaler = StandardScaler()
    # test_imputed_std = pd.DataFrame(std_scaler.fit_transform(test_imputed), columns=test_imputed.columns)

    # discretize necessary object columns and concat to df
    dumb5 = pd.get_dummies(df_test['x5'], prefix='x5', prefix_sep='_', dummy_na=True)
    test_imputed = pd.concat([test_imputed, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df_test['x31'], prefix='x31', prefix_sep='_', dummy_na=True)
    test_imputed = pd.concat([test_imputed, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df_test['x81'], prefix='x81', prefix_sep='_', dummy_na=True)
    test_imputed = pd.concat([test_imputed, dumb81], axis=1, sort=False)

    # set missing features to 0
    fin = set(features)
    fih = set(list(test_imputed.columns))

    # missing features not in data
    fm = list(fin - fih)

    test_imputed[fm] = 0

    test_imputed[fm] = test_imputed[fm].astype(np.int)

    df_test = test_imputed[features]

    # pass params through model and return result
    df_preds = pd.DataFrame(model.predict(df_test.values), columns=['probs'])
    df_preds['y'] = np.zeros(df_preds.shape[0], dtype=int)
    df_preds.loc[df_preds.probs >= .75, 'y'] = 1

    result = dict(business_outcome=int(df_preds.y.values[0]), phat=float(df_preds.probs.values[0]), params=df_test.to_dict('records'))

    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return "<h1> I'm here </h1>"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', type=str, required=True, help='ip address of device')
    parser.add_argument('-port', type=int, required=True, help='port number to serve api on')
    args = parser.parse_args()

    # start app
    app.run(host=args.ip, port=args.port, debug=True)

