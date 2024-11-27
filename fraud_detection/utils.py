import requests
import math
from decouple import config

def get_coordinates(city_name):
    """Fetch latitude and longitude of a city using OpenCage Geocoder."""
    API_KEY = config('OPENCAGE_API_KEY')  # Fetch from .env
    base_url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': city_name,
        'key': API_KEY,
        'limit': 1,
        'countrycode': 'ca'  # Limit results to Canada
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            geometry = data['results'][0]['geometry']
            return geometry['lat'], geometry['lng']
        else:
            raise ValueError(f"City '{city_name}' not found.")
    else:
        raise ConnectionError("Failed to connect to the geocoding API.")

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth."""
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
