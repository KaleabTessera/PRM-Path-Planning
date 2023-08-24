from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse
from scipy.stats import qmc
from .dijkstra import Graph, dijkstra, to_array
from .utils import Utils
import time

class Prmcontroller:
    def __init__(self, num_of_random_coordinates, all_obs, current, destinations):
        self.num_of_coords = num_of_random_coordinates
        self.coords_list = np.array([])
        self.all_obs = all_obs
        self.current = np.array(current)
        self.destinations = np.array(destinations)
        self.graph = Graph()
        self.utils = Utils()
        self.solution_found = False
        self.max_size_of_map = 100

    def run_prm(self, initial_random_seed, save_image=True):
        seed = initial_random_seed
        start = time.time()
        while not self.solution_found:
            print("Trying with random seed {}".format(seed))
            np.random.seed(seed)
            sampler = qmc.Halton(d=2, scramble=False, seed=seed)
            sample = sampler.random(n=self.num_of_coords)
            self.coords_list = sample * self.max_size_of_map
            for i in range(0, len(self.destinations)):
                if i != 0:
                    self.current = self.destinations[i - 1, :]
                self.destination = self.destinations[i, :]
                self.generate_coords()
                self.check_if_collision_free()
                self.find_nearest_neighbour()
                self.shortest_path()

            seed = np.random.randint(1, 100000)
            self.coords_list = np.array([])
            self.graph = Graph()

            if save_image:
                plt.savefig("{}_samples.png".format(self.num_of_coords))
            plt.show()
            end = time.time()
            print(f"{end - start:.5f} sec")

    def generate_coords(self):
        self.current = self.current.reshape(1, 2)
        self.destination = self.destination.reshape(-1, 2)
        self.coords_list = np.concatenate(
            (self.coords_list, self.current, self.destination), axis=0)

    def check_if_collision_free(self):
        collision = False
        self.collision_free_points = np.array([])
        for point in self.coords_list:
            collision = self.check_point_collision(point)
            if not collision:
                if self.collision_free_points.size == 0:
                    self.collision_free_points = point
                else:
                    self.collision_free_points = np.vstack(
                        [self.collision_free_points, point])
        self.plot_points(self.collision_free_points)

    def find_nearest_neighbour(self, k=8):
        X = self.collision_free_points
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collision_free_paths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                if not self.check_point_collision(start_line) and not self.check_point_collision(end_line):
                    if not self.check_line_collision(start_line, end_line):
                        self.collision_free_paths = np.concatenate(
                            (self.collision_free_paths, p.reshape(1, 2), neighbour.reshape(1, 2)), axis=0)

                        a = str(self.find_node_index(p))
                        b = str(self.find_node_index(neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j + 1])
                        x = [p[0], neighbour[0]]
                        y = [p[1], neighbour[1]]
                        plt.plot(x, y)

    def shortest_path(self):
        self.start_node = str(self.find_node_index(self.current))
        self.end_node = str(self.find_node_index(self.destination))

        dist, prev = dijkstra(self.graph, self.start_node)
        path_to_end = to_array(prev, self.end_node)

        if len(path_to_end) > 1:
            self.solution_found = True
        else:
            return

        points_to_display = [(self.find_points_from_node(path))
                             for path in path_to_end]

        x = [int(item[0]) for item in points_to_display]
        y = [int(item[1]) for item in points_to_display]
        plt.plot(x, y, c="blue", linewidth=3.5)

        points_to_end = [str(self.find_points_from_node(path))
                          for path in path_to_end]
        print("****Output****")
        print("The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
            self.collision_free_points[int(self.start_node)],
            self.collision_free_points[int(self.end_node)],
            " \n ".join(points_to_end),
            str(dist[self.end_node])
        ))

    def check_line_collision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.all_obs:
            if self.utils.is_wall(obs):
                unique_coords = np.unique(obs.all_coordinates, axis=0)
                wall = shapely.geometry.LineString(
                    unique_coords)
                if line.intersection(wall):
                    collision = True
            else:
                obstacle_shape = shapely.geometry.Polygon(obs.all_coordinates)
                collision = line.intersects(obstacle_shape)
            if collision:
                return True
        return False

    def find_node_index(self, p):
        return np.where((self.collision_free_points == p).all(axis=1))[0][0]

    def find_points_from_node(self, n):
        return self.collision_free_points[int(n)]

    def plot_points(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=1)

    def check_collision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if obs.bottom_left[0] <= p_x <= obs.bottom_right[0] and obs.bottom_left[1] <= p_y <= obs.top_left[1]:
            return True
        else:
            return False

    def check_point_collision(self, point):
        for obs in self.all_obs:
            collision = self.check_collision(obs, point)
            if collision:
                return True
        return False

        return False
