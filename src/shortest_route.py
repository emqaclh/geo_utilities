from typing import Dict, List, Optional

import osmnx as ox
import numpy as np
import networkx as nx

from pyrosm import OSM

from sklearn.neighbors import BallTree
from sklearn.metrics import pairwise

TRAVEL_TYPES = {
    "driving": "driving",
    "cycling": "cycling",
    "walking": "walking",
    "service": "driving+service",
}


class ShortestRoute:

    graphs: Dict = dict()

    def __init__(
        self, osm: OSM, travel_types: List = list(TRAVEL_TYPES.keys())
    ) -> None:
        self.validate_dependencies()
        if osm is not None:
            for travel_type in TRAVEL_TYPES.keys():
                if travel_type not in travel_types:
                    continue
                network_type = TRAVEL_TYPES[travel_type]
                nodes, edges = osm.get_network(nodes=True, network_type=network_type)
                graph = osm.to_graph(nodes, edges, graph_type="networkx")
                graph = ox.add_edge_speeds(graph)
                graph = ox.add_edge_travel_times(graph)
                graph = nx.convert_node_labels_to_integers(graph)
                self.graphs[travel_type] = graph

    def validate_dependencies(self) -> None:
        if pairwise.distance_metrics is None:
            raise Warning(
                "There is no valid distance metrics available on the current scikit-learn installation."
            )

    def get_valid_distance_metrics(self) -> List[str]:
        return pairwise.distance_metrics()

    def validate_distance_metric(self, distance_metric: str) -> None:
        if distance_metric not in pairwise.distance_metrics():
            raise Warning(
                f"{distance_metric} is not a valid distance metric. Please use one of the following: {self.get_valid_distance_metrics}"
            )

    def get_available_travel_types(self) -> List:
        return list(self.graphs.keys())

    def validate_travel_type(self, travel_type: str = "driving") -> None:
        if travel_type not in self.get_available_travel_types():
            raise Warning(
                f"Travel type is not available. Available travel types are: {self.get_available_travel_types()}"
            )

    def get_nodes_positions(self, travel_type: str = "driving") -> np.ndarray:
        self.validate_travel_type(travel_type)
        graph = self.graphs[travel_type]
        node_pos = np.zeros((graph.number_of_nodes(), 2))
        for node, data in graph.nodes(data=True):
            node_pos[node, [0, 1]] = data["x"], data["y"]
        return node_pos

    def build_nodes_tree(
        self, travel_type: str = "driving", distance_metric: str = "haversine"
    ):
        self.validate_travel_type(travel_type)
        self.validate_distance_metric(distance_metric)
        nodes = self.get_nodes_positions(travel_type)
        tree = BallTree(nodes, metric=distance_metric)
        return tree

    def get_nearest_node(
        self,
        x: float,
        y: float,
        travel_type: str = "driving",
        return_dist: bool = False,
    ):
        self.validate_travel_type(travel_type)
        return ox.distance.nearest_nodes(
            self.graphs[travel_type], x, y, return_dist=return_dist
        )

    def get_nearest_nodes(
        self,
        sources: np.ndarray,
        travel_type: str = "driving",
        return_dist: bool = False,
        tree: Optional[BallTree] = None
    ):
        if tree is None:
            tree = self.build_nodes_tree(travel_type)
        if return_dist:
            d, i = tree.query(sources, k=1, return_distance=return_dist)
            return d[:, 0], i[:, 0]
        else:
            i = tree.query(sources, k=1, return_distance=return_dist)
            return i[:, 0]

    def compute_cost_matrix(self, sources: np.ndarray, targets: np.ndarray, travel_type: str = 'driving', weight: str = 'travel_time'):
        self.validate_travel_type(travel_type)
        graph = self.graphs[travel_type]
        tree = self.build_nodes_tree(travel_type)
        source_nodes = self.get_nearest_nodes(sources, travel_type=travel_type, return_dist=False, tree=tree)
        target_nodes = self.get_nearest_nodes(targets, travel_type=travel_type, return_dist=False, tree=tree)
        cost_matrix = np.full( (len(sources), len(targets)), np.inf )
        for index_source, source in enumerate(source_nodes):
            distances = nx.shortest_path_length(graph, source=source, weight=weight)
            for index_target, target in enumerate(target_nodes):
                cost_matrix[index_source, index_target] = distances[target]
        return cost_matrix
    
    def compute_travel_time_matrix(self, sources: np.ndarray, targets: np.ndarray, travel_type: str = 'driving'):
        return self.compute_cost_matrix(sources, targets, travel_type=travel_type, weight='travel_time')
    
    def compute_distance_matrix(self, sources: np.ndarray, targets: np.ndarray, travel_type: str = 'driving'):
        return self.compute_cost_matrix(sources, targets, travel_type=travel_type, weight='length')