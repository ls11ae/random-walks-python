from pyproj import Transformer
import math

# Beispiel-Bounding Box in Lon/Lat (Deutschland)
bbox_latlon = (6.146663, 45.709082, 13.877637, 52.751451)  # (min_lon, min_lat, max_lon, max_lat)

print("BBox in Lon/Lat:", bbox_latlon)

# 1. Transformiere Bounding Box in UTM (flächentreu)
#    Wir nehmen automatisch passende UTM-Zone für den Mittelpunkt
mid_lon = (bbox_latlon[0] + bbox_latlon[2]) / 2
utm_zone = int((mid_lon + 180) / 6) + 1
utm_epsg = 32600 + utm_zone  # Nordhalbkugel

transformer_to_utm = Transformer.from_crs("EPSG:3035", f"EPSG:{utm_epsg}", always_xy=True)
min_x, min_y = transformer_to_utm.transform(bbox_latlon[0], bbox_latlon[1])
max_x, max_y = transformer_to_utm.transform(bbox_latlon[2], bbox_latlon[3])
bbox_utm = (min_x, min_y, max_x, max_y)
print("BBox in UTM (Meter):", bbox_utm)

# 2. Berechne Aspect Ratio
width_m = max_x - min_x
height_m = max_y - min_y
aspect_ratio = width_m / height_m
print("Aspect ratio (width / height):", aspect_ratio)

# 3. Setze gewünschte Resolution auf der längeren Achse
resolution_longer = 600  # z.B. 600 Zellen auf der längeren Achse

if width_m >= height_m:
    res_x = resolution_longer
    res_y = max(1, int(resolution_longer / aspect_ratio))
else:
    res_y = resolution_longer
    res_x = max(1, int(resolution_longer * aspect_ratio))
print("Grid size (res_x, res_y):", (res_x, res_y))

# 4. Erzeuge Grid-Koordinaten in UTM
#    Wir wollen 0..res_x/0..res_y, jeweils mit Flächenäquivalent
grid_points_utm = []
for iy in range(res_y):
    for ix in range(res_x):
        # UTM Koordinaten
        utm_x = min_x + (ix / (res_x - 1)) * (max_x - min_x)
        utm_y = max_y - (iy / (res_y - 1)) * (max_y - min_y)  # y invertiert, damit obere linke Ecke (0,0)
        grid_points_utm.append((utm_x, utm_y))

# 5. Optional: zurück zu Lon/Lat
transformer_to_latlon = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:3035", always_xy=True)
grid_points_latlon = [transformer_to_latlon.transform(x, y) for x, y in grid_points_utm]

# Testausgabe einiger Punkte
print("Beispielpunkte in Lon/Lat:")
for p in grid_points_latlon[::len(grid_points_latlon)//5]:  # 5 Punkte als Beispiel
    print(p)
