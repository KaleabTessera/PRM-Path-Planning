import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry


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
            # if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
            plt.scatter(x, y, c="red")
            plt.plot(x, y)
        else:
            currentAxis.add_patch(Rectangle(
                (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

    plt.scatter(curr[0], curr[1], c='green')
    plt.scatter(dest[0], dest[1], c='green')
    # fig = plt.figure()
    # plt.show()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    print(data.shape)
    # data = np.fromstring(fig, dtype=np.uint8, sep='')
    # plt.show()


class PRMController:
    def __init__(self, numOfRandomCoordinates, allObs, current, destination):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.array([])
        self.allObs = allObs
        self.current = current
        self.destination = destination

    def runPRM(self):
        self.genCoords()
        self.checkIfCollisonFree()
        self.findNearestNeighbour()

    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            if(isWall(obs)):
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    uniqueCords)
                if(line.intersection(wall)):
                    print(wall, line, line.intersection(wall))
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacleShape)
            if(collision):
                return True
        return False

    def findNearestNeighbour(self):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        print(indices[0])
        for i, p in enumerate(X):
            # print(p, indices[i])
            for i in X[indices[i]]:
                start_line = [p[0], i[0]]
                end_line = [p[1], i[1]]
                if(not self.checkPointCollision(start_line) and not self.checkPointCollision(end_line)):
                    # try:
                    if(not self.checkLineCollision(start_line, end_line)):
                            # self.checkLineCollision(line)
                        plt.plot(start_line, end_line)
                    # except:
                        # print(start_line, end_line, Exception)

    def genCoords(self):
        self.coordsList = np.random.randint(
            100, size=(self.numOfCoords, 2))

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
            # print("No collision", point)
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

    prm = PRMController(1000, allObs, current, destination)
    prm.runPRM()

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
