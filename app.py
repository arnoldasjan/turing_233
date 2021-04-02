from flask import Flask, request
import pickle
import json
import numpy as np


app = Flask(__name__)

SAVED_MODEL_PATH = "classifier.pkl"

classifier = pickle.load(open(SAVED_MODEL_PATH, "rb"))


@app.route('/')
def hello_world():
    return 'Hello World!'


def __process_input(request_data: str) -> np.array:
    parsed_body = np.asarray(json.loads(request_data)["inputs"])
    assert len(parsed_body.shape) == 2, "'inputs' must be a 2-d array"
    return parsed_body


@app.route("/predict", methods=["POST"])
def predict() -> str:
    try:
        input_params = __process_input(request.data)
        predictions = classifier.predict(input_params)

        return json.dumps({"predicted_class": predictions.tolist()})
    except (KeyError, json.JSONDecodeError, AssertionError):
        return json.dumps({"error": "CHECK INPUT"}), 400
    except:
        return json.dumps({"error": "PREDICTION FAILED"}), 500


if __name__ == '__main__':
    app.run()
