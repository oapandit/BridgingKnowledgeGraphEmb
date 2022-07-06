# from sklearn.base import BaseEstimator
# from sklearn.exceptions import NotFittedError
import numpy as np


# class IRFF(BaseEstimator):
class IRFF():
    '''
    Random fourier features using the improved embedding
    https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
    '''
    def __init__(self, gamma=1., n_components=None):
        self.gamma = gamma
        self.n_components = n_components
        self.fitted = False

    def fit(self, X, y=None):
        inp = np.array(X)
        d = inp.shape[-1]
        if self.n_components is None:
            self.n_components = inp.shape[-1]
        D = int(self.n_components/2)

        self.w = np.sqrt(2*self.gamma)*np.random.normal(size=(D, d))

        self.fitted = True
        return self

    def transform(self, X):
        inp = np.array(X)
        if not self.fitted:
            raise NotFittedError('Fourier feature should be fitted before transforming')

        # dotproduct = inp.dot(self.w.T)
        print("caldualting dot p")
        dotproduct = np.matmul(inp, self.w.T)
        print("dot p shape {}".format(dotproduct.shape))
        Z = np.sqrt(2 / self.n_components) * np.concatenate([np.cos(dotproduct), np.sin(dotproduct)], axis=-1)
        return Z

    def fit_transform(self, X):
        self.fit(X)
        print("fitted x")
        return self.transform(X)

    def compute_kernel(self, X):
        if not self.fitted:
            raise NotFittedError('Fourier feature should be fitted before computing kernel')

        Z = self.transform(X)
        return Z.dot(Z.T)
        # return np.matmul(Z, self.w.T)
