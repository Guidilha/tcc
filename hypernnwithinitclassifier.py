from math import tau
import torch
import time
import numpy as np
from src.models.abstractclassifier import AbstractClassifier
from src.models.hypernnclassifier import HyperNNClassifier
#from hypernn.src.utils import calcular_comprimentos_topologicos_som 
from minisom import MiniSom
from sklearn.cluster import KMeans, DBSCAN

class HyperNNWithInitClassifier(AbstractClassifier):
    def __init__(self,
                 nboxes=2,
                 init_strategy='random' ,
                 som_sigma=1.0,
                 som_lr=0.5,
                 gamma_escala=1.0,
                 dbscan_eps=0.5,
                 dbscan_min_samples=5,
                 tau=1.0,
                 alpha=1.0,
                 alpha_tau_decay_step=10,
                 device='cuda',
                 l1_reg=1e-3,
                 l2_reg=1e-3,
                 training_epochs=1000,
                 random_state=42,
                 learning_rate=1e-1,
                 patience_early_stopping=200):


        self.nboxes = nboxes
        self.init_strategy = init_strategy
        self.som_sigma = som_sigma
        self.som_lr = som_lr
        self.gamma_escala = gamma_escala
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state
        self.tau = tau
        self.alpha = alpha
        self.alpha_tau_decay_step = alpha_tau_decay_step
        self.device = device
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.patience_early_stopping = patience_early_stopping


        self.clf = None

    def _calcular_comprimentos_topologicos_som(self, som_weights_3d,x,y,dim, gamma=None):
      comprimentos = np.zeros((x*y,dim))
      min_len = 0.15

      for i in range(x):
        for j in range(y):
          idx_linear = i*y+j
          vizinhos_pesos = []

          if i > 0:
            vizinhos_pesos.append(som_weights_3d[i-1,j])
          if i < x-1:
            vizinhos_pesos.append(som_weights_3d[i+1,j])
          if j > 0:
            vizinhos_pesos.append(som_weights_3d[i,j-1])
          if j < y-1:
            vizinhos_pesos.append(som_weights_3d[i,j+1])

          if len(vizinhos_pesos) == 0:
            comprimentos[idx_linear] = min_len
          else:
            diffs = [np.abs(som_weights_3d[i,j]-v) for v in vizinhos_pesos]
            comprimentos[idx_linear] = np.maximum(self.gamma_escala * np.mean(diffs,axis=0),min_len)

      return torch.from_numpy(comprimentos).float()

    def _calculate_kmeans_init(self, X_target):
        n_samples, n_features = X_target.shape
        n_clusters = min(self.nboxes, n_samples)
        
        kmeans= KMeans(n_clusters=n_clusters, random_state=getattr(self, 'seed', 42), n_init='auto')
        labels = kmeans.fit_predict(X_target)
        
        mins_list = []
        lengths_list = []

        for k in range(self.nboxes):
            if k < n_clusters:
               cluster_points = X_target[labels == k]
               
               if len(cluster_points) > 1:
                   c_min_real = np.min(cluster_points, axis=0) 
                   c_max_real = np.max(cluster_points, axis=0) 
                   center = (c_min_real + c_max_real) / 2.0 
                   raw_len = (c_max_real - c_min_real) * self.gamma_escala 
                   c_len = np.maximum(raw_len, 1e-3)
                   c_min = center - (c_len / 2.0)
               elif len(cluster_points) == 1:
                  center = cluster_points[0]
                  c_len = np.full(n_features, 0.1 * self.gamma_escala) 
                  c_min = center - (c_len / 2.0)
               else:
                   center = kmeans.cluster_centers_[k]
                   c_len = np.full(n_features, 0.1* self.gamma_escala)
                   c_min = center - (c_len/2.0)
            else:
                c_min = np.zeros(n_features)
                c_len = np.ones(n_features) * 0.1
            
            mins_list.append(c_min)
            lengths_list.append(c_len)

        mins_tensor = torch.tensor(np.array(mins_list), dtype=torch.float32)
        lengths_tensor = torch.tensor(np.array(lengths_list), dtype=torch.float32)
        
        return mins_tensor, lengths_tensor
        
    def _calculate_som_init(self, X_som): 
        n_samples, dim = X_som.shape 
        x_grid = 1 
        y_grid = self.nboxes
         
        som = MiniSom(x=x_grid, y=y_grid, input_len=dim, sigma=self.som_sigma, learning_rate=self.som_lr)
        som.random_weights_init(X_som)
        som.train_random(data=X_som, num_iteration=2000)
           
        som_weights_3d = som.get_weights()
        shaped_weights = som_weights_3d.reshape(-1,dim)
        centers_tensor = torch.from_numpy(shaped_weights).float()
               
        lengths_tensor = self._calcular_comprimentos_topologicos_som(
            som_weights_3d=som_weights_3d,
            x=x_grid, y=y_grid, dim=dim, gamma=self.gamma_escala
        )
        mins_tensor = centers_tensor - (lengths_tensor / 2.0)
        return mins_tensor, lengths_tensor
     
    def _calculate_dbscan_init(self, X_target: np.ndarray):
      dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
      labels = dbscan.fit_predict(X_target)
      unique_labels = [lbl for lbl in set(labels) if lbl != -1]  
 
      if len(unique_labels) > self.nboxes: 
        counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels} 
        sorted_labels = sorted( counts.keys(), key=lambda lbl: counts[lbl], reverse=True ) 
        unique_labels = sorted_labels[:self.nboxes] 
      
      theta_m_list = [] 
      theta_l_list = [] 
   
      for lbl in unique_labels: 
        cluster_points = X_target[labels == lbl] 
        min_pt = np.min(cluster_points, axis=0) 
        max_pt = np.max(cluster_points, axis=0)  
        length = np.maximum(max_pt - min_pt, 1e-3) 
         
        theta_m_list.append(min_pt) 
        theta_l_list.append(length) 
        
      mins_tensor = torch.tensor(np.array(theta_m_list), dtype=torch.float32) 
      lengths_tensor = torch.tensor(np.array(theta_l_list), dtype=torch.float32) 
      
      return mins_tensor, lengths_tensor 


    def fit(self, X, y):
        self.dim = X.shape[1]
        t0_init = time.time()

        self.clf = HyperNNClassifier(
            nboxes=self.nboxes,
            dim=self.dim, # Fixed: dim to self.dim
            tau=self.tau,
            alpha=self.alpha,
            alpha_tau_decay_step=self.alpha_tau_decay_step,
            device=self.device,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            training_epochs=self.training_epochs,
            learning_rate=self.learning_rate,
            patience_early_stopping=self.patience_early_stopping,
        )

        X_target = X[y==1]

        if self.init_strategy == 'som' and len(X_target) > 0:
          mins_tensor, lengths_tensor = self._calculate_som_init(X_target)
          with torch.no_grad():
            self.clf.model.block.mins.copy_(mins_tensor)
            self.clf.model.block.length.copy_(lengths_tensor)
          self.clf.model.block.mins.requires_grad = True

        elif self.init_strategy == 'kmeans' and len(X_target) >= self.nboxes:
          mins_tensor, lengths_tensor = self._calculate_kmeans_init(X_target)
          with torch.no_grad(): 
              self.clf.model.block.mins.copy_(mins_tensor.to(self.device)) 
              self.clf.model.block.length.copy_(lengths_tensor.to(self.device))
           
        elif self.init_strategy == "dbscan" and len(X_target) > 0: 
          mins_tensor, lengths_tensor = self._calculate_dbscan_init(X_target) 
          if mins_tensor is not None and lengths_tensor is not None: 
            with torch.no_grad(): 
              self.clf.model.block.mins.copy_(mins_tensor.to(self.device)) 
              self.clf.model.block.length.copy_(lengths_tensor.to(self.device))

        elif self.init_strategy == 'random':
          pass
        
        self.init_time_ = time.time() - t0_init

        self.clf.fit(X,y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def set_hyperparameters(self, hyperparameters:dict):
        return HyperNNWithInitClassifier(**hyperparameters)

    def get_num_boxes(self):
      lengths = self.clf.model.block.length if hasattr(self,'clf') else self.model.block.length
      dim = self.clf.dim if hasattr(self, 'clf') else self.dim

      return torch.sum(torch.abs(lengths) > 1e-3).item()/(self.dim)