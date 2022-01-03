from Point import Point

class Node:
    def __init__(self, value=None):
        self.element = value
        self.left = None
        self.right = None

    def distance_1d(self, P, axis):
        new_values = P.values.copy()
        new_values[axis] = self.element

        p = Point(new_values)

        return p.distance(P)