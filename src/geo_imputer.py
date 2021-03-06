from typing import List

import numpy as np
import geopandas as gpd

from sklearn.neighbors import BallTree
from sklearn.metrics import pairwise

from shapely.geometry import Point


class GeoImputer:

    geodf = None
    tree = None

    def __init__(self, geodf: gpd.GeoDataFrame) -> None:
        if isinstance(geodf, gpd.GeoDataFrame):
            geodf.reset_index(inplace=True, drop=True)
            self.geodf = geodf
            self.tree = self.build_tree()
        else:
            raise TypeError("Must provide a valid GeoDataFrame object")

    def impute(self, points: np.ndarray, column: str, k=5):
        if not self.column_exists(column):
            raise Exception(f"{column} does not exists.")
        distances, indexes = self.tree.query(points, k=k)
        containing_matrix = self.get_containing_matrix(points, indexes)
        col_values = np.full(indexes.shape, np.nan)
        result = np.full(col_values.shape[0], np.nan)

        for i, index in enumerate(indexes):
            col_values[i, :] = self.get_column_values_per_index(column, index)

        imput_index = np.where(
            containing_matrix.sum(axis=1) > 0, containing_matrix.argmax(axis=1), -1
        )
        for i, imput_i in enumerate(imput_index):
            if imput_i > -1:
                result[i] = col_values[i][imput_i]
            else:
                result[i] = np.average(col_values[i], weights=distances[i])
        return result

    def get_column_values_per_index(
        self, column: str, indexes: np.ndarray
    ) -> np.ndarray:
        return self.geodf.loc[indexes, column]

    def get_containing_matrix(
        self, points: np.ndarray, indexes: np.ndarray
    ) -> np.ndarray:
        result_matrix = np.full(indexes.shape, False)
        for i, point in enumerate(points):
            geometry_point = Point(point)
            geometries = self.geodf.loc[indexes[i], "geometry"]
            result_matrix[i, :] = geometries.contains(geometry_point).values
        return result_matrix

    def build_tree(self, distance_metric: str = "haversine") -> BallTree:
        self.validate_distance_metric(distance_metric)
        coords = self.get_geocoordinates()
        tree = BallTree(coords, metric=distance_metric)
        return tree

    def get_geocoordinates(self, representative_point=True) -> np.ndarray:
        points = (
            self.geodf.representative_point()
            if representative_point
            else self.geodf.centroid
        )
        positions = np.transpose(np.array([points.x, points.y]))
        return positions

    def column_exists(self, column_name: str) -> bool:
        return column_name in self.geodf.columns

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
