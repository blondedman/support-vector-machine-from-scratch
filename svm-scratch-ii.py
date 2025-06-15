import numpy as np
import pandas as pd

class svm:
    
    def __init__(self, visualization=False):
        
        self.visualization = visualization
        self.w = None
        self.b = None

    def fit(self, data):
        
        self.data = data
        
        key = list(self.data.keys())[0]
        dim = self.data[key].shape[1]
        
        all_data = []
        opt_dict = {}
        
        for yi in self.data:
            for features in self.data[yi]:
                all_data.extend(features)
                
        self.max_feat_val = max(all_data)
        self.min_feat_val = min(all_data)
        
        all_data = None
        
        step_sizes = [self.max_feat_val * 0.1,
                      self.max_feat_val * 0.01,
                      self.max_feat_val * 0.001]
        
        b_multiple = 5
        b_range_multiple = 5
        
        latest_optimum = self.max_feat_val * 10
        
        transforms = [np.ones(dim)]
        
        for step in step_sizes:
            w = np.ones(dim) * latest_optimum
            optimized = False
            while not optimized:
                for b in np.arange(-1 * self.max_feat_val * b_range_multiple,
                                   self.max_feat_val * b_range_multiple,
                                   step * b_multiple):
                    for t in transforms:
                        wt = w * t
                        found_option = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi * (np.dot(wt, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        
                        if found_option:
                            opt_dict[np.linalg.norm(wt)] = [wt, b]
                
                if w[0] < 0:
                    optimized = True
                
                else:
                    w = w - step
            
            norms = sorted([n for n in opt_dict])
            
            if norms:
                opt_choice = opt_dict[norms[0]]
                
                # ||w|| = [w,b]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                
                latest_optimum = opt_choice[0][0] + step * 2
                
            else:
                # no solution found for this step
                pass

        # Minimal Fix: check if a solution was found before using self.w/self.b
        if self.w is None or self.b is None:
            print("no separating hyperplane found")
            return

        # (optional) print margin values for each sample
        # for yi in self.data:
        #     for xi in self.data[yi]:
        #         print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        
        if self.w is None or self.b is None:
            raise ValueError("model has not been fitted with a separating hyperplane.")
        
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        
        return classification

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
df = df.apply(pd.to_numeric)

X = np.array(df.drop(['class'], axis=1)).astype(float)
y = np.array(df['class']).astype(int)

# class labels to -1 and 1
y_binary = np.where(y == 2, -1, 1)

data_dict = {-1: [], 1: []}

for features, label in zip(X, y_binary):
    data_dict[label].append(features)

data_dict[1] = np.array(data_dict[1])
data_dict[-1] = np.array(data_dict[-1])

SVM = svm(visualization=False)
SVM.fit(data_dict)

if SVM.w is not None and SVM.b is not None:
    print("prediction for sample 0 is", SVM.predict(X[0]))
else:
    print("brute forcw SVM could not find a separating hyperplane")