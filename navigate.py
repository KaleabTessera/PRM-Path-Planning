from collections import defaultdict
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse


class Navigate:
    def __init__(self, mobileObject):
        self.mobileObject = mobileObject


class Obstacle:
    def __init__(self, topLeft, bottomRight):
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.calcFullCord()

    def printFullCords(self):
        print(self.topLeft, self.topRight, self.bottomLeft, self.bottomRight)

    def calcFullCord(self):
        otherP1 = [self.topLeft[0], self.bottomRight[1]]
        otherP2 = [self.bottomRight[0], self.topLeft[1]]

        points = [self.topLeft, otherP1,
                  otherP2, self.bottomRight]

        # Finding correct coords and what part of rectangle they represent - we can't always assume we receive the top left and bottomRight
        x = [item[0] for item in points]
        y = [item[1] for item in points]

        minX = np.min(x)
        minY = np.min(y)

        maxX = np.max(x)
        maxY = np.max(y)

        self.topRight = np.array([maxX, maxY])
        self.bottomLeft = np.array([minX, minY])

        self.topLeft = np.array([minX, maxY])
        self.bottomRight = np.array([maxX, minY])

        self.allCords = [self.topLeft, self.topRight,
                         self.bottomLeft, self.bottomRight]

        self.width = self.topRight[0] - self.topLeft[0]
        self.height = self.topRight[1] - self.bottomRight[1]


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        # self.distances[(to_node, from_node)] = distance


def isWall(obs):
    x = [item[0] for item in obs.allCords]
    y = [item[1] for item in obs.allCords]
    if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
        return True  # Wall
    else:
        return False  # Rectangle


def drawMap(obs, curr, dest):
    fig = plt.figure()
    currentAxis = plt.gca()
    for ob in obs:
        if(isWall(ob)):
            x = [item[0] for item in ob.allCords]
            y = [item[1] for item in ob.allCords]
            plt.scatter(x, y, c="red")
            plt.plot(x, y)
        else:
            currentAxis.add_patch(Rectangle(
                (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

    plt.scatter(curr[0], curr[1], s=200, c='green')
    plt.scatter(dest[0], dest[1], s=200, c='green')
    fig.canvas.draw()


class PRMController:
    def __init__(self, numOfRandomCoordinates, allObs, current, destination):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.array([])
        self.allObs = allObs
        self.current = np.array(current)
        self.destination = np.array(destination)
        self.graph = Graph()

    def runPRM(self):
        self.genCoords()
        self.checkIfCollisonFree()
        self.findNearestNeighbour()
        self.startNode = self.findNodeIndex(self.current)
        self.endNode = self.findNodeIndex(self.destination)
        self.findShortestPath(self.graph, self.startNode)
        for key, value in self.graph.distances.items():
            if(key[1] == self.endNode):
                print(key, self.collisionFreePoints[
                    key[0]], self.collisionFreePoints[key[1]], value)
        # print(self.graph.distances.keys.value)

    # Dijkstra

    def findShortestPath(self, graph, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(graph.nodes)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in graph.edges[min_node]:
                try:
                    weight = current_weight + graph.distances[(min_node, edge)]
                except:
                    weight = current_weight + math.inf
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node
                    if(edge == self.endNode):
                        print("HMMM", self.collisionFreePoints[min_node])

        return visited, path

    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            if(isWall(obs)):
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    uniqueCords)
                if(line.intersection(wall)):
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacleShape)
            if(collision):
                return True
        return False

    def findNodeIndex(self, p):
        return np.where(self.collisionFreePoints == p)[0][1]

    def findNearestNeighbour(self):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)
        # print(indices[0])

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                if(not self.checkPointCollision(start_line) and not self.checkPointCollision(end_line)):
                    if(not self.checkLineCollision(start_line, end_line)):
                        self.collisionFreePaths = np.concatenate(
                            (self.collisionFreePaths, p.reshape(1, 2), neighbour.reshape(1, 2)), axis=0)
                        self.graph.add_node(self.findNodeIndex(p))
                        self.graph.add_edge(self.findNodeIndex(
                            p), self.findNodeIndex(neighbour), distances[i, j+1])
                        x = [p[0], neighbour[0]]
                        y = [p[1], neighbour[1]]
                        plt.plot(x, y)
                        # print(f'found neigh: {p} and {neighbour}')
                        # break
        # return collisionFreePaths

    def genCoords(self):

        self.coordsList = np.random.randint(
            100, size=(self.numOfCoords, 2))
        # Adding begin and end points
        self.current = self.current.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.coordsList = np.concatenate(
            (self.coordsList, self.current, self.destination), axis=0)
        # print(self.coordsList)
        # self.coordsList.append()

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=1)

    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]):
            return True
        else:
            return False

    def checkPointCollision(self, point):
        for obs in self.allObs:
            collision = self.checkCollision(obs, point)
            if(collision):
                return True
        return False

    def checkIfCollisonFree(self):
        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            collision = self.checkPointCollision(point)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])
        self.plotPoints(self.collisionFreePoints)


def main(args):
    graph = Graph()
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--numSamples', type=int, default=100, metavar='N',
                        help='Number of sampled points')
    args = parser.parse_args()

    numSamples = args.numSamples

    env = open("environment.txt", "r")
    l1 = env.readline().split(";")

    current = list(map(int, l1[0].split(",")))
    destination = list(map(int, l1[1].split(",")))

    print("Current: {} Destination: {}".format(current, destination))

    print("****Obstacles****")
    allObs = []
    for l in env:
        if(";" in l):
            line = l.strip().split(";")
            topLeft = list(map(int, line[0].split(",")))
            bottomRight = list(map(int, line[1].split(",")))
            obs = Obstacle(topLeft, bottomRight)
            obs.printFullCords()
            allObs.append(obs)

    drawMap(allObs, current, destination)

    prm = PRMController(numSamples, allObs, current, destination)
    prm.runPRM()

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
