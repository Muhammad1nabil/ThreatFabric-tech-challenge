import json
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/predict', methods=['POST'])
def predict_user():
    data = request.get_json()
    model_name = data['Model']
    if model_name.upper() not in ['SVM', 'RF', 'XGB']:
        return jsonify(
            msg=f'invalid model, ({model_name})',
            status=400)

    model = pickle.load(open(f'{model_name.lower()}.model', 'rb'))

    record = np.array([data['HT']['Mean'], data['PPT']['Mean'], data['RPT']['Mean'], data['RRT']['Mean'],
                       data['HT']['STD'], data['PPT']['STD'], data['RPT']['STD'], data['RRT']['STD']])

    prediction = model.predict(record.reshape(1, 8))
    print(prediction)
    return jsonify({'user': int(prediction)})


if __name__ == '__main__':
    app.run()
