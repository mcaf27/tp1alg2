from Point import Point
from KDTree import KDTree
from random import random
import sys
import re

data = []
n_variables = 0
classes = []

file_name = sys.argv[1]

with open(file_name, 'r') as f:
    lines = f.readlines()
    index = 0
    for i, line in enumerate(lines):
        if line.startswith('@inputs'):
            classes = re.compile('(.*){(.*)}').split(lines[i-1].strip())[2].split(', ')
            if len(classes) == 1:
                classes = classes[0].split(',')
        elif line.startswith('@data\n'):
            index = i + 1
            break
    
    data = lines[index:]

TRAIN_TEST_SPLIT = 0.7

points = []
train = []
test = []

for item in data:
    item = (item.strip('\n')).split(',')
    p = Point([float(x) for x in item[:-1]], item[-1].strip())
    if random() >= TRAIN_TEST_SPLIT:
        train.append(p)
    else:
        test.append(p)

# assumindo que nenhuma entrada dos dados vai ter dados faltando
n_variables = len(train[0].values)

kdtree = KDTree(n_variables)
root = kdtree.build(train, 0)

k = 3

mat = dict(zip(classes, (dict(zip(classes, [0] * len(classes))) for _ in range(len(classes)))))

for item in test:
    p = Point(item.values)
    pred = kdtree.k_nearest_neighbors(root, p, k, classes)

    mat[pred][item.label] += 1

for key in mat:
    print(key, mat[key])

c1, c2 = classes
tp = mat[c1][c1]
tn = mat[c2][c2]
fp = mat[c1][c2]
fn = mat[c2][c1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'precisão: {precision:.4f} / revocação: {recall:.4f}')