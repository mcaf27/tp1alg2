import numpy as np

class LimitedPriorityQueue:
    def __init__(self, max_size):
        self.queue = [] # Points
        self.max_size = max_size

    def __repr__(self):
        return f'{self.queue}'

    def largest_distance(self, P):
        if len(self.queue) > 0:
            return self.queue[-1].distance(P)
        else:
            return np.inf
    
    def smallest_distance(self, P):
        if len(self.queue) > 0:
            return self.queue[0].distance(P)
        else:
            return np.ninf

    def insert(self, item, comp):
        esq = 0
        dir = len(self.queue)
        
        while esq < dir:
            mid = (esq + dir) // 2
            if item.distance(comp) < self.queue[mid].distance(comp):
                dir = mid
            else:
                esq = mid + 1
        
        self.queue.insert(esq, item)

        if len(self.queue) > self.max_size:
            self.queue.pop()