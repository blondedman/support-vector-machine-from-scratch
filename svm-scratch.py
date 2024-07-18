import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class svm:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    def fit(self, data):
        self.data = data
        
        
        transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]
        
        opt_data = {}
        all_data = []
        
        for yi in self.data:
            for features in self.data[yi]:
                for feature in features:
                    all_data.append(feature)
                    
        self.max_feat_val =  max(all_data)
        self.min_feat_val =  min(all_data)
        all_data = None
        
        step_size = [self.max_feat_val * 0.1,
                     self.max_feat_val * 0.01,
                     self.max_feat_val * 0.001] # point of expense
        
        # extremely expensive
        b_range_mutilple = 5
        
        b_multiple = 5
        
        latest_opt = self.max_feat_val * 10
        
        for step in step_size:
            w = np.array([latest_opt,latest_opt])
            optimized = False  
            while not optimized:
                pass
    
    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        
        return classification

data = {-1: np.array([[1,7],[2,8],[3,8]]),
         1: np.array([[5,1],[6,-1],[7,3]])}