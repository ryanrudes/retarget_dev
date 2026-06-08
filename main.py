import orjson
from rich.pretty import pprint as print
from pathlib import Path

unbagged_path = Path("bags/ground_estimation/unbagged")

with open(unbagged_path / "tf.json", "r") as file:
    tf = orjson.loads(file.read())

with open(unbagged_path / "vicon_markers.json", "r") as file:
    vicon_markers = orjson.loads(file.read())

tf_times = set()
for datetime, message in tf.items():
    header = message["header"]
    stamp = header["stamp"]
    time = stamp["sec"] + stamp["nanosec"] * 1e-9
    tf_times.add(time)

for datetime, message in vicon_markers.items():
    header = message["header"]
    stamp = header["stamp"]
    time = stamp["sec"] + stamp["nanosec"] * 1e-9
    # if time not in tf_times:
    #     print(datetime)
    print(message)