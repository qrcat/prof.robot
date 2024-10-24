import json
import pathlib
import collections



data_path = pathlib.Path("data/unitree_h1")

iteration = []
collision = []
distances = []
for json_path in data_path.glob('*/info.json'):
    with json_path.open() as f:
        data = json.load(f)
    iteration.append(data['iter'])
    collision.append(data['collision'])
    distances.append(data.get('distance', 0))

collision_counter = collections.Counter(collision)
