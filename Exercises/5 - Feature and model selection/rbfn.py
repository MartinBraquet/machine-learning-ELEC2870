import numpy as np
from competitive_learning import get_inits, comp_learning, show_quantization
from linear_model import MyLinearRegressor, compute_rmse

class MyRBFN():
    def __init__(self, nb_centers, width_scaling):
        super().__init__()
        self.nb_centers = nb_centers
        self.width_scaling = width_scaling
        
        self.linear_model = MyLinearRegressor(add_bias = True)
        
    def fit_centers(self,X):
        centroid_inits = get_inits(X,self.nb_centers)
        # c is of shape (nb_centers,X.shape[1])
        self.c = comp_learning(X, centroid_inits, n_epochs=100, alpha=0.1, beta=0.99, min_epsilon=1e-3)
    
    def fit_widths(self,X):
        distances = np.sqrt(np.sum((np.repeat(X, self.nb_centers, axis=0).reshape(X.shape[0], self.nb_centers, X.shape[1]) - self.c)**2, axis=-1))
        closest_center = np.argmin(distances, axis=-1)
        self.s = []
        for center in range(self.nb_centers):
            center_samples = np.where(closest_center == center)
            self.s.append(np.mean(distances[center_samples,center]))
        self.s = np.array(self.s) * self.width_scaling
        
    def fit_weights(self,X,y):
        self.linear_model.fit(self.non_linear_transform(X),y)
        
    def fit(self, X, y):
        self.fit_centers(X)
        self.fit_widths(X)
        self.fit_weights(X,y)
    
    def non_linear_transform(self,X):
        '''
        Applies the non-linear transformation to the inputs
        '''
        out = np.ndarray((X.shape[0],self.nb_centers))
        for i in range(self.nb_centers):
            out[:,i] = np.exp( - 1./2. * np.sum((X-self.c[i])**2,axis =1)/(self.s[i]**2+1e-7))
        return out
    
    def predict(self, X):
        return self.linear_model.predict(self.non_linear_transform(X))
    
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
    
    def score(self, X, y_true):
        y = self.predict(X)
        return compute_rmse(y, y_true)