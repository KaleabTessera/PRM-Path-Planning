import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

class Utils:
    def is_wall(self, obs):
        x = [item[0] for item in obs.all_coordinates]
        y = [item[1] for item in obs.all_coordinates]
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return True  # Wall
        else:
            return False  # Rectangle

    def draw_map(self, obs, curr, dest):
        fig = plt.figure()
        current_axis = plt.gca()
        for ob in obs:
            if self.is_wall(ob):
                x = [item[0] for item in ob.all_coordinates]
                y = [item[1] for item in ob.all_coordinates]
                plt.scatter(x, y, c="red")
                plt.plot(x, y)
            else:
                current_axis.add_patch(Rectangle(
                    (ob.bottom_left[0], ob.bottom_left[1]), ob.width, ob.height, alpha=0.4))
        dest = np.array(dest)
        plt.scatter(curr[0], curr[1], s=200, c='green')
        plt.scatter(dest[:, 0], dest[:, 1], s=200, c='green')
        fig.canvas.draw()
