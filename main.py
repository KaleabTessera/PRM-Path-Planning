
import sys
import numpy as np
import argparse
from classes import PRMController, Obstacle, Utils
import time

def main(args):

    parser = argparse.ArgumentParser(description='PRM Path Planning Algorithm')
    parser.add_argument('--numSamples', type=float, default=1000, metavar='N',
                        help='Number of sampled points')
    args = parser.parse_args()

    numSamples = args.numSamples

    env = open("environment.txt", "r")
    # l1 : Current and Destination
    l1 = env.readline().split(";")

    current = list(map(float, l1[0].split(",")))
    destinations = np.array(list(map(float, l1[1].split(",")))).reshape(-1,2)
    destinations = np.ndarray.tolist(destinations)

    print("Current: {} Destinations: {}".format(current, destinations))

    print("****Obstacles****")
    # 모든 장애물들
    allObs = []
    for l in env:
        if(";" in l):
            line = l.strip().split(";")
            topLeft = list(map(int, line[0].split(",")))
            bottomRight = list(map(int, line[1].split(",")))
            obs = Obstacle(topLeft, bottomRight)
            obs.printFullCords()
            allObs.append(obs)

    utils = Utils()
    utils.drawMap(allObs, current, destinations)

    prm = PRMController(numSamples, allObs, current, destinations)
    # Initial random seed to try
    initialRandomSeed = 27
    prm.runPRM(initialRandomSeed)


if __name__ == '__main__':
    # prm의 결과가 이제 총 경로의 길이가 나오도록 코드 수정.
    main(sys.argv)
    
