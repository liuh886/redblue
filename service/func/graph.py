import numpy as np
import osmnx as ox
from shapely.geometry import Point, LineString
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point

class GraphBeacon:
    def __init__(self, graph, beacons):
        """
        Initializes the GraphBeacon class.

        :param graph: An OSMnx graph representing the railway network.
        :param beacons: A dictionary or list of beacons with their locations and metadata.
        """
        self.graph = graph
        self.beacons = beacons
        self.train_locations = {}  # Dictionary to hold train locations on the graph

    def add_beacon(self, beacon_id, location, metadata=None):
        """
        Adds a new beacon to the beacons dictionary.

        :param beacon_id: Unique identifier for the beacon.
        :param location: Tuple of (latitude, longitude) representing the beacon's location.
        :param metadata: Optional dictionary of additional beacon metadata.
        """
        self.beacons[beacon_id] = {'location': location, 'metadata': metadata}

    def update_train_location(self, train_id, node=None, edge=None, distance_from_node=None):
        """
        Updates the location of a train on the graph.

        :param train_id: Unique identifier for the train.
        :param node: The graph node closest to the train's current location.
        :param edge: The graph edge the train is currently on.
        :param distance_from_node: Distance from the specified node along the edge.
        """
        self.train_locations[train_id] = {'node': node, 'edge': edge, 'distance_from_node': distance_from_node}

    def get_train_location(self, train_id):
        """
        Retrieves the current location of a train on the graph.

        :param train_id: Unique identifier for the train.
        :return: A dictionary containing the train's location information.
        """
        return self.train_locations.get(train_id, None)

    def process_beacon_detection(self, train_id, beacon_id):
        """
        Processes the detection of a beacon by a train, updating the train's location.

        :param train_id: Unique identifier for the train.
        :param beacon_id: Unique identifier for the detected beacon.
        """
        beacon_location = self.beacons[beacon_id]['location']
        # Logic to find the nearest node or edge to the beacon location and update the train's location
        pass

    def map_coordinates_to_graph(self, coordinates, heading=None):
        """
        Maps real-world coordinates to the nearest node or edge on the graph, considering the heading.

        :param coordinates: Tuple of (latitude, longitude).
        :param heading: Optional heading direction in degrees.
        :return: Nearest node or edge and distance to the node.
        """
        pass

    # Additional methods as needed for integration with the Kalman Filter, querying the graph, etc.

def convert_1D_to_2D(G, current_node, nearest_edge, distance_from_node, heading):

    # Determine the direction of the edge
    direction = determine_travel_direction(G, current_node, nearest_edge, heading)

    if direction == 'towards':
        distance = distance_from_node
    else:
        distance = nearest_edge.length - distance_from_node

    # Handle negative distance by finding a preceding edge
    if distance < 0:
        # Find a preceding edge
        preceding_edge, remaining_distance = find_preceding_edge(G, nearest_edge, abs(distance), heading)
        if preceding_edge is None:
            # Handle case where no preceding edge is found
            return None  # Or some other error handling
        point_on_edge = G.edges[preceding_edge]['geometry'].interpolate(remaining_distance)
    
    # Handle distance exceeding the edge length by finding a subsequent edge
    elif distance > G.edges[nearest_edge]['length']:
        # Find a subsequent edge
        subsequent_edge, remaining_distance = find_subsequent_edge(G, nearest_edge, distance - G.edges[nearest_edge]['length'], heading)
        if subsequent_edge is None:
            # Handle case where no subsequent edge is found
            return None  # Or some other error handling
        point_on_edge = G.edges[subsequent_edge]['geometry'].interpolate(remaining_distance)

    else:
        # The point is on the edge
        point_on_edge = nearest_edge.interpolate(distance)

    return point_on_edge.x, point_on_edge.y

def find_preceding_edge(G, current_node, distance, heading):
    best_edge = None
    min_angle_diff = 360  # Maximum angle difference
    for u, v, data in G.in_edges(current_node, data=True):
        if 'geometry' in data:
            line = data['geometry']
            first_point = line.coords[0]
            last_point = line.coords[-1]
        else:
            first_point = (G.nodes[u]['x'], G.nodes[u]['y'])
            last_point = (G.nodes[v]['x'], G.nodes[v]['y'])

        edge_bearing = calculate_bearing(last_point, first_point)  # Note the reversed order for preceding edge
        angle_diff = min(abs(edge_bearing - heading), 360 - abs(edge_bearing - heading))

        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_edge = (u, v, data)

    if best_edge is None:
        return None, distance

    u, v, data = best_edge
    edge_length = data['length']
    if edge_length >= distance:
        return (u, v), distance
    else:
        return find_preceding_edge(G, u, distance - edge_length, heading)


def find_subsequent_edge(G, current_node, distance, heading):
    best_edge = None
    min_angle_diff = 360  # Maximum angle difference
    for u, v, data in G.out_edges(current_node, data=True):
        if 'geometry' in data:
            line = data['geometry']
            first_point = line.coords[0]
            last_point = line.coords[-1]
        else:
            first_point = (G.nodes[u]['x'], G.nodes[u]['y'])
            last_point = (G.nodes[v]['x'], G.nodes[v]['y'])

        edge_bearing = calculate_bearing(first_point, last_point)
        angle_diff = min(abs(edge_bearing - heading), 360 - abs(edge_bearing - heading))

        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_edge = (u, v, data)

    if best_edge is None:
        return None, distance

    u, v, data = best_edge
    edge_length = data['length']
    if edge_length >= distance:
        return (u, v), distance
    else:
        return find_subsequent_edge(G, v, distance - edge_length, heading)


def convert_2D_to_1D(G, point_coords, heading):
    # Point representing the current position
    point = Point(point_coords)
    
    # Find the nearest edge to the point
    nearest_edge = ox.distance.nearest_edge(G, point_coords)
    edge_geom = LineString(G[nearest_edge[0]][nearest_edge[1]][0]['geometry'])
    # Project the point onto the edge to find the exact point on the railway track
    projected_point = edge_geom.interpolate(edge_geom.project(point))
    
    # Calculate distances to the start and end nodes of the edge
    start_node_geom = Point(G.nodes[nearest_edge[0]]['x'], G.nodes[nearest_edge[0]]['y'])
    end_node_geom = Point(G.nodes[nearest_edge[1]]['x'], G.nodes[nearest_edge[1]]['y'])
    dist_to_start_node = projected_point.distance(start_node_geom)
    dist_to_end_node = projected_point.distance(end_node_geom)
    
    # Determine the direction of the edge
    edge_direction = np.degrees(np.arctan2(end_node_geom.y - start_node_geom.y, end_node_geom.x - start_node_geom.x)) % 360
    
    # Determine the nearest node and distance based on heading
    if abs((heading - edge_direction) % 360) < 180:
        # Moving towards the end node
        nearest_node = nearest_edge[1]
        dist_to_nearest_node = dist_to_end_node
        dist_from_nearest_node = dist_to_start_node
    else:
        # Moving towards the start node
        nearest_node = nearest_edge[0]
        dist_to_nearest_node = dist_to_start_node
        dist_from_nearest_node = dist_to_end_node
    
    return nearest_node, nearest_edge, dist_to_nearest_node, dist_from_nearest_node

def calculate_bearing(pointA, pointB):
    """
    Calculate the bearing between two points A and B.

    Parameters:
    - pointA: Tuple of (longitude, latitude) for point A.
    - pointB: Tuple of (longitude, latitude) for point B.

    Returns:
    - Bearing in degrees from point A to point B.
    """
    lon1, lat1 = np.radians(pointA[0]), np.radians(pointA[1])
    lon2, lat2 = np.radians(pointB[0]), np.radians(pointB[1])
    dLon = lon2 - lon1

    x = np.cos(lat2) * np.sin(dLon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)

    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360  # Normalize to 0-360

    return bearing


def determine_travel_direction(G, current_point, nearest_edge, heading):
    """
    Determine the travel direction based on the current position and heading.

    Parameters:
    - G: The graph of the area.
    - current_position: A tuple representing the current position (latitude, longitude).
    - heading: The current heading in degrees from north.

    Returns:
    - direction: A string indicating the travel direction ('towards' or 'away from' the nearest node based on the heading).
    """
    
    # Get the orientation of the nearest edge
    u, v, key = nearest_edge
    point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
    point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
    line = LineString([point_u, point_v])
    edge_orientation = ox.bearing.get_bearing(point_u, point_v)

    # Compare the heading with the edge orientation to determine the direction
    orientation_difference = (heading - edge_orientation + 360) % 360
    if orientation_difference < 180:
        direction = 'towards'
    else:
        direction = 'away from'

    return direction
