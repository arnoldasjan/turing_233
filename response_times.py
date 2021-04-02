import requests
import json
from input import pixels_list


def avg_response_times_individually():
    resp_times = []
    for inp in pixels_list:
        resp = requests.post("http://127.0.0.1:5000/predict",
                             data=json.dumps({"inputs": [inp]}))
        resp_times.append(resp.elapsed.total_seconds())
        print(json.loads(resp.text))
        print(resp.elapsed.total_seconds())

    return sum(resp_times) / len(resp_times)


def response_time_batch():
    resp = requests.post("http://127.0.0.1:5000/predict",
                         data=json.dumps({"inputs": pixels_list}))
    print(json.loads(resp.text))

    return resp.elapsed.total_seconds()


avg_times_individually = avg_response_times_individually()
batch_time = response_time_batch()

print(avg_times_individually)
print(batch_time)

print(f"Individual requests are  {batch_time / avg_times_individually:.2f} times faster than batch request")