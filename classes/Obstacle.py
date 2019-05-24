import numpy as np


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
