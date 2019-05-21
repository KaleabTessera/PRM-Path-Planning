import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np


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
        self.bottomLeft = [self.topLeft[0], self.bottomRight[1]]
        self.topRight = [self.bottomRight[0], self.topLeft[1]]
        self.width = self.topRight[0] - self.topLeft[0]
        self.height = self.topRight[1] - self.bottomRight[1]
        self.allCords = [self.topLeft, self.topRight,
                         self.bottomLeft, self.bottomRight]


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

    def genCoords(self):
        self.coordsList = np.random.randint(
            100, size=(self.numOfCoords, 2))
        # x = [item[0] for item in self.coordsList]
        # y = [item[1] for item in self.coordsList]
        # plt.scatter(x, y, c="black", s=1)
        # plt.show()

    def plotPoints(self,points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=1)

    def checkCollision(self,obs,point):
        p_x = point[0]
        p_y = point[1]
        #Check collision
        if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.topLeft[1] <= p_y <= obs.bottomLeft[1]):
            return True
        return False

    def checkIfCollisonFree(self):

        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            for obs in self.allObs:
                collision = self.checkCollision(obs,point)
                if(collision):
                    continue
            if(self.collisionFreePoints.size == 0):
                self.collisionFreePoints = point
            else:
                self.collisionFreePoints = np.vstack([self.collisionFreePoints,point])
        # print(self.collisionFreePoints)
        # print(self.collisionFreePoints.shape)
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

    prm = PRMController(100, allObs, current, destination)
    prm.runPRM()

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
