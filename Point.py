class Point:
    def __init__(self, values, label=None):
        self.values = values
        self.label = label

    def __repr__(self):
        return f'{self.values} || {self.label}'

    def __getitem__(self, i):
        return self.values[i]

    def distance(self, P):
        d = 0
        for v1, v2 in zip(self.values, P.values):
            d += (v1 - v2)**2
        return d