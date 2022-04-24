from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='')
app.config['DEBUG'] = False


@app.route('/predict', methods=['POST'])
def predict_user():
    try:
        # reading request json
        data = request.get_json()
        # getting 'Model' from request to select which model gonna be used and load the corresponding pickle model
        model_name = data['Model']
        if model_name.upper() not in ['SVM', 'RF', 'XGB']:
            return jsonify(msg=f'invalid model, {model_name}'), 400

        model = pickle.load(open(f'{model_name.lower()}.model', 'rb'))

        # creating np array with shape (1, 8) aligned correctly like the training model
        record = np.array([data['HT']['Mean'], data['PPT']['Mean'], data['RPT']['Mean'], data['RRT']['Mean'],
                           data['HT']['STD'], data['PPT']['STD'], data['RPT']['STD'], data['RRT']['STD']]).reshape(1, 8)

        prediction = model.predict(record)
        # response payload has the predicted user ID
        return jsonify({'user': int(prediction)})

    except ValueError:
        return jsonify(
            msg=f'invalid data, {data}'), 400
    except Exception as e:
        return jsonify(
            msg=f'unexpected error, {e}'), 400

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    # return 'hello world!'

if __name__ == '__main__':
    app.run(host='0.0.0.0')