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


def drawMap(obs, curr, dest):
    currentAxis = plt.gca()
    for ob in obs:
        x = [item[0] for item in ob.allCords]
        y = [item[1] for item in ob.allCords]
        if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
            plt.scatter(x, y, c="red")
            plt.plot(x, y)
        else:
            currentAxis.add_patch(Rectangle(
                (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

    plt.scatter(curr[0], curr[1], c='green')
    plt.scatter(dest[0], dest[1], c='green')
    plt.show()


def main(args):
    env = open("environment-3.txt", "r")
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


if __name__ == '__main__':
    main(sys.argv)
