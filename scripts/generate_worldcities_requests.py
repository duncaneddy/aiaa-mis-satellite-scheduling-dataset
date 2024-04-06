import pathlib
import math
import json
import csv
import collections

from brahe.constants import R_EARTH
from brahe.epoch import Epoch
from brahe.data_models import Request

# Output Directory
OUTPUT_DIR = pathlib.Path('outputs')

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# Set Simulation Duration
window_open = Epoch(2020, 1, 1, time_system='UTC')
DAYS = 7
window_close = window_open + 86400 * DAYS

LOOK_ANGLE_MAX = 55

# Data Types
Location = collections.namedtuple('Location', ['country', 'city', 'lon', 'lat', 'population'])

# Load Requests
locations = []

with open('./data/decks/worldcities.csv') as fp:
    csv_reader = csv.DictReader(fp)
    line_count = 0

    for row in csv_reader:

        country = row['country']
        city = row['city_ascii']
        lat = row['lat']
        lon = row['lng']
        population = row['population']

        if len(row['population']) == 0:
            population == '0.0'

        try:
            locations.append(
                Location(country, city, float(lon), float(lat), int(population))
            )
        except Exception as e:
            locations.append(
                Location(country, city, float(lon), float(lat), 0)
            )

        line_count += 1

locations.sort(key=lambda x: x.population, reverse=True)

# Generate requests
requests = []
for loc in locations:
    request_json = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [loc.lon, loc.lat]
        },
        "properties": {
            "t_start": window_open.isoformat(),
            "t_end": window_close.isoformat(),
            "description": f"{loc.city} - {loc.country} - Population: {loc.population:d}"
        }
    }

    requests.append(Request(**request_json))

NUM_REQUESTS = len(locations)
requests = list(requests[0:min(NUM_REQUESTS, len(requests))])

print(f'Processed {NUM_REQUESTS} requests.')

# Save to Output
requests_out = []
for request in requests:
    requests_out.append(json.loads(request.json()))
json.dump(requests_out, open( OUTPUT_DIR / 'requests.json', 'w'), indent=4)