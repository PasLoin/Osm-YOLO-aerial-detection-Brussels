#### Kaggle notebook configuration ####

!pip install ultralytics
!pip install pyproj
!pip install geojson
!pip install shapely

import os
import requests
from PIL import Image
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, ElementTree
from ultralytics import YOLO
from pyproj import Transformer
import json
import gc
import geojson
from shapely.geometry import Polygon, Point
import math
#!wget https://raw.githubusercontent.com/PasLoin/Osm-YOLO-aerial-detection-Brussels/main/models/zebra/Zebra_crossing-v1.pt
##/kaggle/input/zebra-1-1/pytorch/default/1/Zebra-1-1.pt
# Load your trained model
model = YOLO('/kaggle/input/zebra-1-1/pytorch/default/1/Zebra-1-1.pt')

# Directory to save JOSM XML, BBOX JSON, and GeoJSON
output_dir = '/kaggle/working/'
os.makedirs(output_dir, exist_ok=True)

# Define the projection for WGS84 (Lat/Lon) and Web Mercator
wgs_proj = 'epsg:4326'  # WGS84 Lat/Lon
web_mercator_proj = 'epsg:3857'  # Web Mercator

# Define the large bounding box in WGS84 (EPSG:4326)
large_bbox_wgs84 = [4.287415,50.842153,4.338312,50.857001]  # Change to your needs

# Convert the large bounding box to Web Mercator
transformer = Transformer.from_crs(wgs_proj, web_mercator_proj, always_xy=True)
large_bbox_web_mercator = transformer.transform_bounds(*large_bbox_wgs84)

# Define the tile size in pixels
tile_size_pixels = 640

# Function to Convert Pixel Coordinates to Lat/Lon using pyproj
def pixel_to_latlon(x_pixel, y_pixel, img_width, img_height, tile_bbox_wgs84):
    min_lon, min_lat, max_lon, max_lat = tile_bbox_wgs84
    lon_per_pixel = (max_lon - min_lon) / img_width
    lat_per_pixel = (max_lat - min_lat) / img_height

    lon = min_lon + x_pixel * lon_per_pixel
    lat = max_lat - y_pixel * lat_per_pixel  # Subtract because y increases downwards in image coordinates

    return lat, lon

# Function to Fetch WMS Tiles and Save Them
def fetch_tiles():
    # Calculate the number of tiles in the x and y directions
    num_tiles_x = int((large_bbox_web_mercator[2] - large_bbox_web_mercator[0]) / 100) + 1
    num_tiles_y = int((large_bbox_web_mercator[3] - large_bbox_web_mercator[1]) / 100) + 1

    tiles = []
    bbox_data = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile_bbox_web_mercator = [
                large_bbox_web_mercator[0] + i * (large_bbox_web_mercator[2] - large_bbox_web_mercator[0]) / num_tiles_x,
                large_bbox_web_mercator[1] + j * (large_bbox_web_mercator[3] - large_bbox_web_mercator[1]) / num_tiles_y,
                large_bbox_web_mercator[0] + (i + 1) * (large_bbox_web_mercator[2] - large_bbox_web_mercator[0]) / num_tiles_x,
                large_bbox_web_mercator[1] + (j + 1) * (large_bbox_web_mercator[3] - large_bbox_web_mercator[1]) / num_tiles_y
            ]
            tiles.append(tile_bbox_web_mercator)

            # Convert the tile bbox back to WGS84
            tile_bbox_wgs84 = transformer.transform_bounds(
                tile_bbox_web_mercator[0], tile_bbox_web_mercator[1],
                tile_bbox_web_mercator[2], tile_bbox_web_mercator[3],
                direction='INVERSE'
            )
            bbox_data.append({
                "tile_index": len(tiles) - 1,
                "bbox_wgs84": tile_bbox_wgs84,
                "bbox_web_mercator": tile_bbox_web_mercator
            })

    # Save the BBOX data to a JSON file
    bbox_file_path = os.path.join(output_dir, 'tiles_bbox.json')
    with open(bbox_file_path, 'w') as f:
        json.dump(bbox_data, f, indent=4)

    return num_tiles_x, num_tiles_y, bbox_data, tiles

# Function to Create JOSM XML
def create_josm_xml(detections):
    root = Element('osm', version="0.6", generator="Ultralytics YOLO")

    for i, detection in enumerate(detections):
        lat, lon = detection['properties']['lat'], detection['properties']['lon']
        node_id = -(i + 1)  # Decreasing IDs

        node = SubElement(root, 'node', lat=str(lat), lon=str(lon), id=str(node_id))
        tag = SubElement(node, 'tag', k="highway", v="crossing")

    tree = ElementTree(root)
    josm_file_path = os.path.join(output_dir, 'detections.osm')
    tree.write(josm_file_path)
    print(f"Saved JOSM XML in {josm_file_path}")

# Function to Fetch Existing OSM Data
def fetch_existing_osm_data(bbox):
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["highway"="crossing"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch OSM data: {response.status_code}, {response.content}")
        return None

# Function to Check if Node Exists
def node_exists(node, existing_nodes, threshold_meters=10):
    R = 6371000  # Earth's radius in meters
    node_point = Point(node['lon'], node['lat'])
    for existing_node in existing_nodes:
        existing_point = Point(existing_node['lon'], existing_node['lat'])
        distance = 2 * R * math.asin(math.sqrt(
            (math.sin((existing_point.y - node_point.y) / 2) ** 2 +
             math.cos(node_point.y) * math.cos(existing_point.y) *
             (math.sin((existing_point.x - node_point.x) / 2) ** 2))
        ))
        if distance < threshold_meters:
            return True
    return False

# Function to Calculate Haversine Distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Function to Process Detected Zebra and Create JOSM XML
def process_detections(num_tiles_x, num_tiles_y, bbox_data, tiles, batch_size=100, threshold_meters=10):
    all_detections = []
    total_batches = (len(tiles) + batch_size - 1) // batch_size

    # Fetch existing OSM data
    existing_osm_data = fetch_existing_osm_data(large_bbox_wgs84)
    existing_nodes = []
    if existing_osm_data:
        for element in existing_osm_data['elements']:
            if element['type'] == 'node':
                existing_nodes.append({'id': element['id'], 'lat': element['lat'], 'lon': element['lon']})

    # Process results and create JOSM XML for each detected zebra
    for batch_start in range(0, len(tiles), batch_size):
        batch_end = min(batch_start + batch_size, len(tiles))
        batch_tiles = tiles[batch_start:batch_end]
        batch_bbox_data = bbox_data[batch_start:batch_end]
        images = []
        for i, tile_bbox in enumerate(batch_tiles):
            params = params_template.copy()
            params['BBOX'] = f"{tile_bbox[0]},{tile_bbox[1]},{tile_bbox[2]},{tile_bbox[3]}"
            params['CRS'] = 'EPSG:3857'  # Use Web Mercator for the request

            # Construct the WMS request URL
            wms_request_url = f"{wms_url}?{ '&'.join([f'{k}={v}' for k, v in params.items()]) }"
            response = requests.get(wms_request_url)
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content))
                    img.load()  # Ensure the image is fully loaded into memory
                    images.append(img)
                except Exception as e:
                    print(f"Failed to load image for tile {batch_start + i}: {e}")
            else:
                print(f"Failed to fetch tile {batch_start + i}: {response.status_code}, {response.content}")

        # Run inference on the batch of images
        results = model.predict(source=images, save=False, conf=0.50)

        for i, result in enumerate(results):
            if not result.boxes:
                continue

            original_img_width, original_img_height = images[i].size

            # Extract the bounding box of the tile
            tile_index = batch_start + i
            tile_bbox_wgs84 = batch_bbox_data[i]["bbox_wgs84"]

            for box in result.boxes:
                try:
                    # Extract bounding box data [x_min, y_min, x_max, y_max, confidence, class_id]
                    x_min, y_min, x_max, y_max, confidence, class_id = box.data[0].tolist()

                    # Calculate the bounding box coordinates in WGS84
                    min_lat, min_lon = pixel_to_latlon(x_min, y_min, original_img_width, original_img_height, tile_bbox_wgs84)
                    max_lat, max_lon = pixel_to_latlon(x_max, y_max, original_img_width, original_img_height, tile_bbox_wgs84)

                    # Create a bounding box polygon
                    bbox_polygon = Polygon([
                        (min_lon, min_lat),
                        (max_lon, min_lat),
                        (max_lon, max_lat),
                        (min_lon, max_lat),
                        (min_lon, min_lat)
                    ])

                    # Check if the node exists in the existing OSM data
                    center_lat = (min_lat + max_lat) / 2
                    center_lon = (min_lon + max_lon) / 2

                    # Print the latitude and longitude of the detected zebra's centroid
                    print(f"Detected zebra centroid: Latitude = {center_lat}, Longitude = {center_lon}")

                    # Calculate Haversine distance to each existing node
                    min_distance = float('inf')
                    for existing_node in existing_nodes:
                        distance = haversine_distance(center_lat, center_lon, existing_node['lat'], existing_node['lon'])
                        if distance < min_distance:
                            min_distance = distance

                    if min_distance < threshold_meters:
                        all_detections.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [min_lon, min_lat],
                                    [max_lon, min_lat],
                                    [max_lon, max_lat],
                                    [min_lon, max_lat],
                                    [min_lon, min_lat]
                                ]]
                            },
                            "properties": {
                                "confidence": confidence,
                                "lat": center_lat,
                                "lon": center_lon,
                                "task_type": "node",
                                "task_id": f"task_{len(all_detections) + 1}",
                                "task_description": "Verify if we can add infos",
                                "task_instructions": "Verify task instruction.",
                                "task_tags": ["traffic_calming", "zebra"],
                                "distance": min_distance
                            }
                        })
                    else:
                        all_detections.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [min_lon, min_lat],
                                    [max_lon, min_lat],
                                    [max_lon, max_lat],
                                    [min_lon, max_lat],
                                    [min_lon, min_lat]
                                ]]
                            },
                            "properties": {
                                "confidence": confidence,
                                "lat": center_lat,
                                "lon": center_lon,
                                "task_type": "node",
                                "task_id": f"task_{len(all_detections) + 1}",
                                "task_description": "Add crossing node",
                                "task_instructions": "Verify the presence of a zebra crossing here.",
                                "task_tags": ["crossing", "crossing marking zebra"],
                                "distance": min_distance
                            }
                        })
                except Exception as e:
                    print(f"Error processing box: {e}")

        # Release memory
        del images
        gc.collect()

        # Print progress
        current_batch = (batch_start // batch_size) + 1
        print(f"Processed batch {current_batch}/{total_batches} ({current_batch / total_batches * 100:.2f}% complete)")

    # Separate detections based on distance threshold
    detections_less_than_10m = [detection for detection in all_detections if detection['properties']['distance'] < 10]
    detections_more_than_10m = [detection for detection in all_detections if detection['properties']['distance'] >= 10]

    # Create the JOSM XML file for detected zebras
    create_josm_xml(all_detections)

    # Create the GeoJSON files for MapRoulette challenge
    create_geojson(detections_less_than_10m, 'detections_less_than_10m.geojson')
    create_geojson(detections_more_than_10m, 'maproulette_detections_more_than_10m.geojson')

# Function to Create GeoJSON File
def create_geojson(detections, file_name):
    feature_collection = geojson.FeatureCollection(detections)
    geojson_file_path = os.path.join(output_dir, file_name)
    with open(geojson_file_path, 'w') as f:
        geojson.dump(feature_collection, f, indent=4)
    print(f"Saved GeoJSON file in {geojson_file_path}")

# WMS URL and parameters
wms_url = "https://geoservices-urbis.irisnet.be/geoserver/urbisgrid/ows"
params_template = {
    'FORMAT': 'image/png',
    'TRANSPARENT': 'FALSE',
    'VERSION': '1.3.0',
    'SERVICE': 'WMS',
    'REQUEST': 'GetMap',
    'LAYERS': 'Ortho',
    'STYLES': '',
    'CRS': 'EPSG:3857',  # Use Web Mercator (EPSG:3857) for the request
    'WIDTH': 640,  # Resolution of the image (adjust for your needs)
    'HEIGHT': 640
}

# Function to Save Overpass Query Result to JSON
def save_overpass_result(result, file_path):
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Saved Overpass query result to {file_path}")

# Run the script
num_tiles_x, num_tiles_y, bbox_data, tiles = fetch_tiles()  # Fetch WMS tiles
existing_osm_data = fetch_existing_osm_data(large_bbox_wgs84)  # Fetch existing OSM data
save_overpass_result(existing_osm_data, os.path.join(output_dir, 'existing_osm_data.json'))  # Save Overpass query result to JSON
process_detections(num_tiles_x, num_tiles_y, bbox_data, tiles, batch_size=100, threshold_meters=10)  # Run YOLO and process results into JOSM XML and GeoJSON
