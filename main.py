from Point import Point
from KDTree import KDTree
from numpy.random import shuffle
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

shuffle(data)

TRAIN_TEST_SPLIT = 0.7

points = []

for item in data:
    item = (item.strip('\n')).split(',')
    p = Point([float(x) for x in item[:-1]], item[-1].strip())
    points.append(p)

split_index = int(len(points)*TRAIN_TEST_SPLIT)
train = points[:split_index]
test = points[split_index:]

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

c1, c2 = classes # c1: "positivo" / c2: "negativo"
tp = mat[c1][c1]
tn = mat[c2][c2]
fp = mat[c1][c2]
fn = mat[c2][c1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f'precisão: {precision} / revocação: {recall} / acurácia: {accuracy}')