import os.path

import folium

def walk_to_osm(walk_coords, animal_id, walk_path):
    start_point = walk_coords[0]
    # m = folium.Map(location=start_point, zoom_start=13, tiles="OpenStreetMap")
    m = folium.Map(location=start_point, zoom_start=13,
               tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attr="Esri")
    folium.PolyLine(walk_coords, color="red", weight=3).add_to(m)
    directory = os.path.join(walk_path, f"{animal_id}_walk_map.html")
    m.save(directory)