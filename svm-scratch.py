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
        
        # steps cannot be threaded
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
                for b in np.arange(self.max_feat_val * b_range_mutilple * -1,
                                   self.max_feat_val * b_range_mutilple,
                                   step * b_multiple):
                    for t in transforms:
                        wt = w * t
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(wt,xi)+b) >= 1:
                                    found_option = False
                                    
                        if found_option:
                            opt_data[np.linalg.norm(wt)] = [wt,b]
                            
                if w[0] < 0:
                    optimized = True
                    print('optimized a step')
                
                else:
                    w = w - step
                    
            norms = sorted([n for n in opt_data])
            opt_choice = opt_data[norms[0]]
            
            # ||w|| = [w,b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            
            latest_opt = opt_choice[0][0] + step * 2 
            
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))
    
    def predict(self, features):
        
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=20, marker="*", c=self.colors[classification])
        return classification

    def visualize(self):
        
        [[self.ax.scatter(x[0], x[1], s=20, color=self.colors[i]) for x in data[i]] for i in data]
        
        # hyperplane = x.w + b
        # v = x.w + b
        def hyperplane(x,w,b,v):
            return (-w[0] * x - b + v ) / w[1]
           
        datarange = (self.min_feat_val * 0.9, self.max_feat_val * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        # +ve hyperplane
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
        
        # -ve hyperplane
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
        
        # boundary
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'k--')
        
        plt.show()
         
data = {-1: np.array([[1,7],[2,8],[3,8]]),
         1: np.array([[5,1],[6,-1],[7,3]])}

SVM = svm()
SVM.fit(data)

predict = [[0,10],[1,3],[3,4],[3,5]]

for p in predict:
    SVM.predict(p)
    
SVM.visualize()