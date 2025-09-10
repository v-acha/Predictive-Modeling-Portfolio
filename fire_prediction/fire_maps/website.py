import geopandas as gpd
import folium
from folium import features
from flask import Flask, render_template
from folium import plugins

# Initialize Flask application
app = Flask(__name__)

import folium
from flask import Flask, render_template
import geopandas as gpd

# Initialize Flask application
app = Flask(__name__)

# Sample data for cities with coordinates (latitude, longitude)
cities = {
    'Los Angeles': {'coords': [34.0522, -118.2437], 'counties': ['Los Angeles', 'Orange', 'Riverside']},
    'San Francisco': {'coords': [37.7749, -122.4194], 'counties': ['San Francisco', 'San Mateo', 'Marin']},
    'San Diego': {'coords': [32.7157, -117.1611], 'counties': ['San Diego', 'Imperial', 'Riverside']}
}

from flask import Flask, render_template
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import random
import json

app = Flask(__name__)

# Generate random fire risk points
def generate_fire_risk_points():
    points = []
    # TODO: replace with call to api that pulls in inferences from the model
    for _ in range(100):
        lat = random.uniform(38, 35)  # Random latitude in California
        lon = random.uniform(-120.5, -118)  # Random longitude in California
        risk_color = random.choice(['#ffcc00', '#ff9900', '#ff3300', '#cc0000'])  # Color based on risk
        points.append({"lat": lat, "lon": lon, "color": risk_color, "risk": random.randint(5, 50)})
    return points

# Route to render the map
@app.route('/')
def default_map_view():
    points = generate_fire_risk_points()
    return render_template('index.html', cities=cities, map_type='default', points=points)

@app.route('/line_map')
def index():
    points = generate_fire_risk_points()
    return render_template('index.html', map_type='line', points=points)

@app.route('/test')
def test_view():
    points = generate_fire_risk_points()
    return render_template('test.html', cities=cities, map_type='pop', points=points)

@app.route('/pop_map')
def pop_map_view():
    points = generate_fire_risk_points()

    return render_template('index.html', cities=cities, map_type='pop', points=points)

if __name__ == '__main__':
    app.run(debug=True)
