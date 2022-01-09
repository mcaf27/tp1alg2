import numpy as np
from Point import Point
from Node import Node
from LimitedPriorityQueue import LimitedPriorityQueue

def median(axis, points):
    P = sorted(points, key=lambda p : p[axis])
    m = P[int(np.floor((len(points)-1)/2))][axis]
    return m

class KDTree:
    def __init__(self, dimension):
        self.dimension = dimension

    def build(self, points, depth):
        p1 = []
        p2 = []
        m = 0
        if len(points) == 1:
            return Node(points[0])
        elif len(points) == 0:
            return
        else:
            axis = depth % self.dimension

            s = sorted(points, key=lambda p: p[axis])
            mid = int(np.floor((len(points)-1)/2))
            m = s[mid][axis]
            p1 = s[:mid+1]
            p2 = s[mid+1:]

            # m = median(axis, points)
            # for point in points:
            #     if point[axis] <= m:
            #         p1.append(point)
            #     else:
            #         p2.append(point)
        
        v_left = self.build(p1, depth + 1)
        v_right = self.build(p2, depth + 1)

        v = Node(m)
        v.left = v_left
        v.right = v_right

        return v

    def knn(self, v, P, queue, depth):
        if v is None:
            return
        elif isinstance(v.element, Point):
            queue.insert(v.element, P)
        else:
            axis = depth % self.dimension
            dist_line = v.distance_1d(P, axis)
            dist_largest = queue.largest_distance(P)
            if dist_largest > dist_line or P[axis] == v.element:
                self.knn(v.left, P, queue, depth + 1)
                self.knn(v.right, P, queue, depth + 1)
            else:
                if P[axis] < v.element:
                    self.knn(v.left, P, queue, depth + 1)
                else:
                    self.knn(v.right, P, queue, depth + 1)

    def k_nearest_neighbors(self, root, P, k, classes):
        queue = LimitedPriorityQueue(k)
        self.knn(root, P, queue, 0)
        d = dict(zip(classes, [0] * len(classes)))
        for item in queue.queue:
            d[item.label] += 1

        pred = max(d, key=lambda p : d[p])
        return pred