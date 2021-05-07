import osmnx as ox
import time
from shapely.geometry import Polygon
import os

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = gdf_edges.index
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

print("osmnx version",ox.__version__)

# Download by a bounding box
bounds = (103.603226,104.088684,1.165715,1.46639)
x1,x2,y1,y2 = bounds
boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
start_time = time.time()
save_graph_shapefile_directional(G, filepath='./network-new')
print("--- %s seconds ---" % (time.time() - start_time))

# Download by place name
# place ="Singapore"
# G = ox.graph_from_place(place, network_type='drive', which_result=2)
# save_graph_shapefile_directional(G, filepath='singapore')

# Download by a boundary polygon in geojson
# import osmnx as ox
# from shapely.geometry import shape
# json_file = open("singapore_boundary.geojson")
# import json
# data = json.load(json_file)
# boundary_polygon = shape(data["features"][0]['geometry'])
# G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
# save_graph_shapefile_directional(G, filepath='Singapore')