# Osm-YOLO-aerial-detection-Brussels

Detection of various object on aerial images using an artificial intelligence model (YOLO) and generates files to facilitate their addition or verification on OpenStreetMap (OSM).

How It Works:

Automatic Detection:
The script uses YOLO (You Only Look Once), a computer vision model, to identify objects in aerial images. YOLO detects objects quickly, providing their location and type. For each object type a custom model is provided.

Verification of Existing Data:
For each detected object, the script checks OpenStreetMap to see if already exists within a 10-meter radius.
If a object is already recorded, the detection is ignored to avoid duplicates.

File Generation:
The script generates a GeoJSON file for tools like MapRoulette, enabling collaborative challenges to validate the detections.
