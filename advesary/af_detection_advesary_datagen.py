import requests
import json
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

MAX_NUM_LAYERS = 5
NUM_MEASUREMENTS = 100000
PROFILE = True


def avg(list):
    return sum(list) / len(list)


def flatten(t):
    return [item for sublist in t for item in sublist]


def send_request_to_network(session, params):
    response = session.post(
        "http://127.0.0.1:8080/predict", json={"input_values": params}
    )
    return json.loads(response.text)


def load_network(session, path):
    response = session.post("http://127.0.0.1:8080/loadmodel", json={"path": path})
    return response.text


def random_parameters_request(session):
    params = [random.uniform(0, 1) for _ in range(4)]
    return_dict = send_request_to_network(session, params)
    return_dict["input_values"] = params
    return return_dict


# Profiles the neural net by performing repeated measurements with different depth.
def profile(session):
    layers_to_timing = {}
    
    activation_f = ["relu", "elu", "softmax", "tanh", "sigmoid"]

    for af in activation_f:
        for layer_cnt in range(1, MAX_NUM_LAYERS + 1):
            # Load new network with layer_cnt layers
            network_loaded = load_network(
                session, f"models/iris_{layer_cnt}_{af}_cross-entropy.onnx"
            )
            tmp = f"{layer_cnt}_{af}"
            layers_to_timing[tmp] = []

            print(f"[+] Loaded Netowork: {network_loaded}")
            # Measure NUM_MEASUREMENTS times with random inputs
            for _ in range(NUM_MEASUREMENTS):
                layers_to_timing[tmp].append(
                    random_parameters_request(session)["prediction_time"]
                )

            # Compute average of measurements
            print(f"    Average timing: {avg(layers_to_timing[tmp])}")

    print("[+] Dumping profiling measurements")
    pickle.dump(layers_to_timing, open("layers_af_timing_dict.pl", "wb"))
    return layers_to_timing

def main():
    # Start a session
    session = requests.session()

    # Generating data for activation function detection
    if PROFILE:
        layers_to_timing = profile(session)

if __name__ == "__main__":
    main()
