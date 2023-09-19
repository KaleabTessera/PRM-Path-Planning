import numpy as np

class Obstacle:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.calculate_full_coordinates()

    def print_full_coordinates(self):
        print(self.top_left, self.top_right, self.bottom_left, self.bottom_right)

    def calculate_full_coordinates(self):
        other_p1 = [self.top_left[0], self.bottom_right[1]]
        other_p2 = [self.bottom_right[0], self.top_left[1]]

        points = [self.top_left, other_p1,
                  other_p2, self.bottom_right]

        x = [item[0] for item in points]
        y = [item[1] for item in points]

        min_x = np.min(x)
        min_y = np.min(y)

        max_x = np.max(x)
        max_y = np.max(y)

        self.top_right = np.array([max_x, max_y])
        self.bottom_left = np.array([min_x, min_y])

        self.top_left = np.array([min_x, max_y])
        self.bottom_right = np.array([max_x, min_y])

        self.all_coordinates = [self.top_left, self.top_right,
                                self.bottom_left, self.bottom_right]

        self.width = self.top_right[0] - self.top_left[0]
        self.height = self.top_right[1] - self.bottom_right[1]

