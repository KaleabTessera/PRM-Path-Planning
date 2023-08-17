import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class Utils:
    def isWall(self, obs):
        x = [item[0] for item in obs.allCords]
        y = [item[1] for item in obs.allCords]
        if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
            return True  # Wall
        else:
            return False  # Rectangle

    def drawMap(self, obs, curr, dest):
        fig = plt.figure()
        currentAxis = plt.gca()
        for ob in obs:
            if(self.isWall(ob)):
                x = [item[0] for item in ob.allCords]
                y = [item[1] for item in ob.allCords]
                plt.scatter(x, y, c="red")
                plt.plot(x, y)
            else:
                currentAxis.add_patch(Rectangle(
                    (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))
        dest = np.array(dest)
        print(dest)
        plt.scatter(curr[0], curr[1], s=200, c='green')
        plt.scatter(dest[:,0], dest[:,1], s=200, c='green')
        fig.canvas.draw()
