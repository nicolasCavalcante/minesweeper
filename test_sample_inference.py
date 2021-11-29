import numpy as np
from itertools import product, combinations
from scipy.linalg import null_space
from collections import defaultdict
from itertools import chain,repeat,count,islice
from collections import Counter
import math

def joint_prob(x: list):
    x = np.array(x)
    y = np.log(x / (1 - x)).sum()
    prob = 1 / (np.exp(-y) + 1)
    return prob

def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i,c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(),counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i,j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i]+1
        for i,j in zip(range(i,r), f(count(j), counts[j:])):
            indices[i] = j



# bin = np.random.randint(2, size=(3000, 16))
H, W = 4, 4
m = H * W
bin = np.random.randint(2 ** m, size=3000)
bin = (((bin[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)

bin = np.array(list(product(*([[0, 1]]*m))))


bin = bin[bin[:, 5] == 1]
bin = bin[bin[:, 10] == 1]
bin = bin[bin[:, [0, 1, 2, 4, 6, 8, 9, 10]].sum(1) == 2]
bin = bin[bin[:, [5, 6, 7, 9, 11, 13, 14, 15]].sum(1) == 3]
bin = bin[bin.sum(1) == 5]
print((bin.sum(0) / bin.shape[0]).reshape(H, W))


H, W = 4, 5
m = H * W

bin = np.array(list(product(*([[0, 1]]*m))))


bin = bin[bin[:, 5] == 1]
bin = bin[bin[:, 10] == 1]
bin = bin[bin[:, [0, 1, 6, 10, 11]].sum(1) == 2]
bin = bin[bin[:, [5, 6, 11, 15, 16]].sum(1) == 3]
bin = bin[bin.sum(1) == 5]
print((bin.sum(0) / bin.shape[0]).reshape(H, W))
print(1)


H, W = 4, 3
m = H * W

bin = np.array(list(product(*([[0, 1]]*m))))


bin = bin[bin[:, 3] == 1]
bin = bin[bin[:, 6] == 1]
bin = bin[bin[:, [0, 1, 4, 6, 7]].sum(1) == 2]
bin = bin[bin[:, [3, 4, 7, 9, 10]].sum(1) == 2]
bin = bin[bin.sum(1) == 6]
ans = (bin.sum(0) / bin.shape[0])
print(ans.reshape(H, W))
print(1)


A = np.zeros((12, 12))
A[0, [0, 1]] = [1, -1]
A[1, [0, 9]] = [1, -1]
A[2, [0, 10]] = [1, -1]
A[3, [2, 5]] = [1, -1]
A[4, [2, 8]] = [1, -1]
A[5, [2, 11]] = [1, -1]
A[6, [4, 7]] = [1, -1]
A[7, :] = 1
A[8, [0, 1, 4, 6, 7]] = 1
A[9, [3, 4, 7, 9, 10]] = 1
A[10, 3] = 1
A[11, 6] = 1
B = np.zeros(12)
B[[0, 1, 2, 3, 4, 5, 6]] = 0
B[7] = 6
B[8] = 2
B[9] = 2
B[10] = 1
B[11] = 1
x = np.linalg.lstsq(A, B)
n = null_space(A)
print(x[0].reshape(H, W))
c = (x[0] * x[0]).sum()
b = 2 * (x[0] * n).sum()
a = (n * n).sum()
r = np.roots([a, b, c])
print(1)
print(A @ np.array([0.375, 0.375, .5625, 1, .125, .5625, 1, .125, .5625, .375, .375, .5625]))

alpha = np.linspace(-.3, .3, 1000)

y = (x[0][:, None] + alpha[None,:] * n) / x[0].sum()

entropy = - (y * np.log2(y)).sum(0)

a = alpha[entropy.argmax()]


tot_mines = 5
H, W = 5, 3
board = np.array([-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1, -1]).reshape(H, W)
print(board)


neig = np.array(list(product([-1, 0, 1], [-1, 0, 1])))
def get_neigs_idx(info, H, W, index):
    neigs = info + neig
    neigs = neigs[(neigs >= 0).all(1)]
    neigs = neigs[(neigs < [H, W]).all(1)]
    return index[neigs[:, 0], neigs[:, 1]]


def call_testsample(mine_ai):
    tot_mines = mine_ai.tot_mines - len(mine_ai.mines)
    H, W = mine_ai.height, mine_ai.width
    board = mine_ai.board

    # tot_mines = 3
    # H, W = 4, 3
    # board = -np.ones((H, W))
    # board[[1, 2], [0, 0]] = [1, 2]

    # tot_mines = 5
    # H, W = 5, 3
    # board = np.array([-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1, -1]).reshape(H, W)

    # brute_probs(board, tot_mines)

    neig = np.array(list(product([-1, 0, 1], [-1, 0, 1])))
    infos = np.array(np.where(board != -1)).T
    index = np.arange((H * W)).reshape(H, W)
    safes_idx = np.array(list(mine_ai.safes))
    if safes_idx.size:
        safes_idx = set(index[safes_idx[:, 0], safes_idx[:, 1]])
    else:
        safes_idx = set()
    mines_idx = np.array(list(mine_ai.mines))
    if mines_idx.size:
        mines_idx = set(index[mines_idx[:, 0], mines_idx[:, 1]])
    else:
        mines_idx = set()
    
    neiglist = dict((k, []) for k in index.ravel())
    for info in infos:
        idx = get_neigs_idx(info, H, W, index)
        for k in idx:
            neiglist[k].append(tuple(info))
    groups = defaultdict(list)
    for key, val in neiglist.items():
        groups[tuple(val)].append(key)
    group_idx2key = defaultdict(list)
    for key, val in enumerate(groups.keys()):
        for v in val:
            group_idx2key[v].append(key)
    
    groups = list(groups.values())
    the_list = [0 for _ in range(board.size)]
    for i, vals in enumerate(groups):
        for v in vals:
            if v in safes_idx:
                the_list[v] = -1
            elif v in mines_idx:
                the_list[v] = -2
            else:
                the_list[v] = i
    iterable=[v for v in the_list if v >= 0]
    if not iterable:
        for i, v in enumerate(the_list):
            the_list[i] = 0 if v==-1 else 1
            ans = np.array(the_list).reshape(H, W)
        return ans
    bomb_combs = combinations_without_repetition(tot_mines, iterable=iterable)
    bomb_combs = np.array(list(bomb_combs))
    # bomb_combs = bomb_combs[(bomb_combs != -1).all(1)]
    for info in infos:
        num_neig = board[info[0], info[1]]
        gidx = group_idx2key[tuple(info)]
        idx = get_neigs_idx(info, H, W, index)
        num_neig -= (np.array(the_list)[idx] == -2).sum()
        bomb_combs = bomb_combs[np.isin(bomb_combs, gidx).sum(1) == num_neig]#TODO:  == num_neig - num of bombs already in the neigbouhood

    grouplens = [len(g) for g in groups]
    comb_probs = []
    cell_probs = []
    for comb in bomb_combs:
        prob = 1
        p_cells = [0 for _ in range(len(groups))]
        for elemen in range(len(groups)):
            num_rep = (comb == elemen).sum()
            nelemen = grouplens[elemen]
            p_cells[elemen] = math.comb(nelemen, num_rep)
            prob *= p_cells[elemen]
            p_cells[elemen] = num_rep / nelemen
        comb_probs.append(prob)
        cell_probs.append(p_cells)
    comb_probs = np.array(comb_probs)
    comb_probs = comb_probs / comb_probs.sum()
    cell_probs = np.array(cell_probs)
    cell_probs = (np.array(cell_probs).T * comb_probs).T
    cell_probs = cell_probs.sum(0)

    for i, v in enumerate(the_list):
        if v == -1:
            the_list[i] = 0
        elif v == -2:
            the_list[i] = 1
        else:
            the_list[i] = cell_probs[v]
    ans = np.array(the_list).reshape(H, W)
    print(ans.round(2))
    return ans


def brute_probs(board, tot_mines):
    H, W = board.shape
    bin = np.array(list(product(*([[0, 1]]*(H * W)))))
    neig = np.array(list(product([-1, 0, 1], [-1, 0, 1])))
    infos = np.array(np.where(board != -1)).T
    index = np.arange((H * W)).reshape(H, W)

    comb = np.array(list(combinations(list(range(H * W)), tot_mines)))
    comb = comb + np.arange(comb.shape[0]).reshape(comb.shape[0], 1) * H * W
    bin = np.zeros(comb.shape[0] *  H * W)
    bin[comb] = 1
    bin = bin.reshape(comb.shape[0],  H * W)
    for info in infos:
        bin = bin[bin[:, index[info[0], info[1]]] == 0]
        num_neig = board[info[0], info[1]]
        neigs = info + neig
        neigs = neigs[(neigs >= 0).all(1)]
        neigs = neigs[(neigs < [H, W]).all(1)]
        idx = index[neigs[:, 0], neigs[:, 1]]
        bin = bin[bin[:, idx].sum(1) == num_neig]
    ans = bin.mean(0)
    print(ans.reshape(H, W))
    return ans
    