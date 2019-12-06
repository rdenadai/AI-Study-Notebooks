import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.neighbors import KDTree


def euclidian(a, b, axis=-1):
    return np.sqrt(np.sum(np.square(a - b), axis=axis))


def calculate_theta(dst, dem):
    return np.mean(np.exp(-((np.square(dst)) / dem)))


def find_neighbourhood(BMU, lattice, radius=5):
    K = np.arange(1.5, 50, 1.5)[radius - 1]
    tree = KDTree(lattice, leaf_size=2)
    all_nn_indices = tree.query_radius([BMU], r=K, return_distance=False)
    pts = np.unique(lattice[all_nn_indices[0]], axis=0)
    obj = np.where((pts[:, 0] == BMU[0]) & (pts[:, 1] == BMU[1]))
    return np.delete(pts, obj[0][0], 0)


def calculate_and_adjust_neighbourhood(xi, W, BMU_node, lr, dem, ng):
    indx = tuple(np.split(ng, 2))
    ngb_node = W[indx]
    dst_n = euclidian(BMU_node, ngb_node, axis=-1)
    theta = calculate_theta(dst_n, dem)
    return indx, W[indx] + (theta * (lr * (xi - ngb_node)))


class SOM:
    def __init__(self, units=15, lr=0.5, radius=5, verbose=False):
        self.units = units
        self._lr = lr
        self._sig = 1
        self._radius = radius
        self.W = np.ones((self.units, self.units, 1)).ravel()
        self.verbose = verbose

    def train(self, x, epochs=200, batch_size=32, iter_decay=25):
        qtd, self._size = x.shape
        # Define the Units Weights
        mn, mx = np.min(x), np.max(x)
        # Create the units (neurons) in the space as a 2D grid
        space = np.linspace(mn, mx, self.units)
        self.W = np.ones((self.units, self.units, self._size)).ravel()
        self.W = self.W.reshape(self.units, self.units, self._size)
        for m in range(self.units):
            for j in range(self._size):
                if j == 0:
                    self.W[:, m, j] = self.W[:, m, j] * space
                elif j == self._size - 1:
                    self.W[:, m, j] = self.W[:, m, j] * space[m]
                else:
                    self.W[:, m, j] = self.W[:, m, j]  # * np.random.uniform(1e-3, 1)
        # Create a lattice matrix to find the Neighbourhood of a BMU
        idx = np.indices(self.W.shape[:2])
        idx = np.vstack([ix.flatten() for ix in idx])
        self._lattice = np.stack([ix for ix in idx], axis=1)
        # Running epochs!
        start = time.time()
        n_samples = int(batch_size / 2)
        mb = np.ceil(x.shape[0] / batch_size).astype(np.int32)
        with ProcessPoolExecutor(max_workers=cpu_count()) as exc:
            for epoch in range(epochs):
                X = shuffle(np.copy(x), replace=True)
                r = 0
                for _ in range(mb):
                    # Mini batch crop
                    ini, end = r * batch_size, (r + 1) * batch_size
                    samples = X[ini:end, :]
                    if len(samples) > n_samples:
                        batch_X = shuffle(samples, n_samples=n_samples)
                    else:
                        batch_X = samples
                    r += 1
                    for xi in batch_X:
                        # Find the BMU of the point
                        dist = euclidian(xi, self.W, axis=-1)
                        BMU = np.asarray(np.where(dist == np.min(dist))).ravel()[:2]
                        # If found a BMU
                        if BMU is not None:
                            # Find Neighbourhood of BMU using lattice matrix
                            ngb = find_neighbourhood(BMU, self._lattice, self._radius)
                            # Neighbourhood update
                            BMU_node = self.W[tuple(BMU)]
                            dem = 2 * (np.square(self._sig))
                            # Multiprocess module
                            fn = partial(
                                calculate_and_adjust_neighbourhood,
                                xi,
                                self.W,
                                BMU_node,
                                self._lr,
                                dem,
                            )
                            for ng in exc.map(fn, ngb, chunksize=10):
                                self.W[ng[0]] = ng[1]
                            # BMU update
                            self.W[tuple(BMU)] += self._lr * (xi - BMU_node)
                if (epoch % iter_decay == 0 and epoch > 0) or epoch == epochs - 1:
                    decay = np.exp(-epoch / epochs)
                    # Learning rate decay
                    self._lr *= decay
                    self._sig = self._radius * decay
                    if self._radius > 1:
                        self._radius = int(np.ceil(self._radius * decay))
                    else:
                        self._radius = 1
                    if self.verbose:
                        print(
                            f"epoch [{epoch}/{epochs}] <=> Running time: {time.time() - start}"
                        )

    def umatrix(self):
        s = self.W.shape[0]
        umatrix = np.zeros((s, s))
        for lat in self._lattice:
            ngb = find_neighbourhood(lat, self._lattice, 1)
            xi = self.W[tuple(lat)]
            for ng in ngb:
                xt = np.array([self.W[tuple(ng)]])
                umatrix[tuple(ng)] += euclidian(xi, xt)
        umatrix = (MinMaxScaler().fit_transform(umatrix) * 255).astype(np.int)
        return np.rot90(np.invert(umatrix))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import axes3d

    scaler = MinMaxScaler()
    x, y = make_blobs(128, n_features=3, centers=3, random_state=42)
    x = scaler.fit_transform(x)

    som = SOM(verbose=True)
    som.train(x)
    W = som.W.flatten().reshape(-1, 3)
    umatrix = som.umatrix()

    s = np.arange(0, som.units)
    x, y = np.meshgrid(s, s)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection="3d")
    plt.title("U-Matrix")
    ax.view_init(80, 45)
    surf = ax.plot_surface(
        x, y, umatrix, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.show()
