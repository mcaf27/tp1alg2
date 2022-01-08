from Point import Point
from KDTree import KDTree
from numpy.random import shuffle
import sys
import re
from time import perf_counter

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

# print(f'treino {len(train)} / {(len(train)*100)/len(data)}%')
# print(f'teste: {len(test)} / {(len(test)*100)/len(data)}%')

# assumindo que nenhuma entrada dos dados vai ter dados faltando
n_variables = len(train[0].values)

t1 = perf_counter()

kdtree = KDTree(n_variables)
root = kdtree.build(train, 0)

time_to_build_tree = perf_counter() - t1

k = 3

mat = dict(zip(classes, (dict(zip(classes, [0] * len(classes))) for _ in range(len(classes)))))

t2 = perf_counter()

for item in test:
    p = Point(item.values)
    pred = kdtree.k_nearest_neighbors(root, p, k, classes)

    mat[pred][item.label] += 1

time_knn = perf_counter() - t2

for key in mat:
    print(key, mat[key])

c1, c2 = classes # c1: "positivo" / c2: "negativo"
tp = mat[c1][c1]
tn = mat[c2][c2]
fp = mat[c1][c2]
fn = mat[c2][c1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'precisão: {precision:.4f} / revocação: {recall:.4f}')

print(f'tempo para construir árvore: {time_to_build_tree:.4f} / tempo para realizar o algoritmo knn: {time_knn:.4f}')
print(f'tempo médio para cada ponto: {(time_knn/len(test)):.4f}')