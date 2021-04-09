from matplotlib.pyplot import close
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import product
from scipy.ndimage.interpolation import rotate, shift
from copy import copy


class InvarianceGenerator(object):
    def __init__(self, overwrite=False):
        self.overwrite = overwrite

        self.labels = None
        self.label_nns = None
        self.X_train = None
        self.y_train = None

        self.nns_cache = {}

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.label_nns = {}
        for i in self.labels:
            X_ = X[y == i].reshape(-1, 784)
            model = NearestNeighbors(n_neighbors=1)
            model.fit(X_)
            self.label_nns[i] = model
        self.X_train = X
        self.y_train = y
    
    def best_neighbor(self, x, y, limits=[3, 3, 30], granularity=[5, 5, 31]):
        key = hash((x.tobytes(), y))
        if key in self.nns_cache:
            return self.nns_cache[key]
        grid = list(product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity))))
        xs = [shift(rotate(x, r, reshape=False), (tx, ty)).reshape(784) for (tx, ty, r) in grid]
        xs = np.asarray(xs.copy())
        
        nns = []
        y_nns = []
        # grids_nn = []
        
        # find a nearest neighbor in each class
        for i in self.labels:
            if i != y:
                X = self.X_train[self.y_train == i]
                Y = self.y_train[self.y_train == i]
                distances, indices = self.label_nns[i].kneighbors(xs, n_neighbors=1)
                best = np.argmin(np.reshape(distances, -1))
                best_idx = np.reshape(indices, -1)[best]
                nns.append(X[best_idx])
                y_nns.append(Y[best_idx])
                
                # store the inverse rotation+translation to be applied to the target
                # grids_nn.append(-np.asarray(grid[best]))
        # return nns, y_nns, grids_nn
        # self.nns_cache[key] = (nns, y_nns)
        return nns, y_nns


    def linf_attack(self, x, nn_adv, eps=0.3):
        x_adv = x.copy().astype(np.float32)
        nn_adv = nn_adv.astype(np.float32)
        
        # if possible, change the pixels to the target value
        idx = np.where((np.abs(nn_adv - x) <= eps*255.) & (x > 0))
        x_adv[idx] = nn_adv[idx]
        
        # otherwise, go as close as possible
        idx = np.where(np.abs(nn_adv - x) > eps*255.)
        sign = np.sign(nn_adv - x)
        x_adv[idx] += sign[idx] * eps * 255.
        
        x_adv = np.clip(x_adv, x.astype(np.float32) - eps*255, x.astype(np.float32) + eps*255)
        x_adv = np.clip(x_adv, 0, 255.)
        
        return x_adv
    
    def invariance_attack(self, x, y, eps=0.3, limits=[3, 3, 30], granularity=[5, 5, 31]):
        nns, y_nns = self.best_neighbor(x, y, limits=limits, granularity=granularity)
        closest = np.inf
        best_x_adv = None
        for nn, y_nn in zip(nns, y_nns):
            x_adv = self.linf_attack(x, nn, eps=eps)
            delta = np.sum(np.abs(np.max(x/255. - x_adv/255., 0)))
            if delta < closest:
                closest = delta
                best_x_adv = x_adv
        return best_x_adv

