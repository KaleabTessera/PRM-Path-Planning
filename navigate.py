#!/usr/bin/env python
import sys
import rospy
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class Navigate:
    def __init__(self,mobileObject):
        self.mobileObject = mobileObject

class Obstacle:
    def __init__(self,topLeft,bottomRight):
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.calcFullCord()

    def printFullCords(self):
        print(self.topLeft,self.topRight,self.bottomLeft,self.bottomRight)

    def calcFullCord(self):
        self.bottomLeft = [self.topLeft[0],self.bottomRight[1]]
        self.topRight = [self.topLeft[1],self.bottomRight[0]]
        self.width = self.topRight[0] - self.topLeft[0]
        self.height = self.topRight[1] - self.bottomRight[1]
    
def drawMap(obs,curr,dest):
    currentAxis = plt.gca()
    for ob in obs:
         currentAxis.add_patch(Rectangle((ob.bottomRight[0], ob.bottomRight[1]), ob.width, ob.height,alpha=0.4))

    plt.scatter(curr[0],curr[1],c='green')
    plt.scatter(dest[0],dest[1],c='green')
    plt.show()

def main(args):
    env = open("environment.txt", "r") 
    l1 = env.readline().split(";")
    
    current = list(map(int,l1[0].split(",")))
    destination = list(map(int,l1[1].split(",")))

    print("Current: {} Destination: {}".format(current,destination))
    
    print("****Obstacles****")
    allObs = []
    for l in env:
        if(";" in l):
            line = l.strip().split(";")
            topLeft = list(map(int,line[0].split(",")))
            bottomRight = list(map(int,line[1].split(",")))
            obs = Obstacle(topLeft,bottomRight)
            obs.printFullCords()
            allObs.append(obs)

    drawMap(allObs,current,destination)

if __name__ == '__main__':
    main(sys.argv)