from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml
from .mldataset import MLDataset

IRIS = 'iris'
SENSIT = 'sensit'
MNIST = 'mnist'
SATIMAGE = 'satimage'
COVTYPE = 'covtype'
LETTER = 'letter'
WINE = 'wine'
CARS = 'cars'
BLOOD = 'blood'
BCANCER = 'breastcancer'
DIGITS = 'digits'
VEHICLE = 'vehicle'
    
class DatasetCollector:    
    def __init__(self):
        self._available_datasets = {
            IRIS: self.load_iris, 
            LETTER: self.load_letter,
            BLOOD: self.load_blood,
            WINE: self.load_wine,
            BCANCER: self.load_breast_cancer,
            CARS: self.load_cars,
            SATIMAGE: self.load_satimage, 
            SENSIT: self.load_sensit, 
            DIGITS: self.load_digits,
            VEHICLE: self.load_vehicle,
            MNIST: self.load_mnist, 
        }
    
    def get_datasets_by_names(self, names:list):
        '''Selects a list of MLDatasets by name'''
        
        datasets = []
        
        for name in names:
            assert name in self._available_datasets.keys(), f'{name} is not in the list of datasets. Available datasets are: {self._available_datasets.keys()}.'
            datasets.append(self._available_datasets[name]())
            
        return datasets
    
    def get_all_datasets(self):
        '''Returns all available MLDatasets'''
       
        return self.get_datasets_by_names(list(self._available_datasets.keys()))
    
    def load_iris(self) -> MLDataset:
        X, y = load_iris(return_X_y=True)
        return MLDataset(X, y, name=IRIS)
    
    def load_sensit(self) -> MLDataset:
        X, y = fetch_openml('SensIT-Vehicle-Combined', version=1, return_X_y=True, as_frame=False)
        X = X.toarray()
        return MLDataset(X, y, name=SENSIT)
            
    def load_mnist(self) -> MLDataset:  
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        return MLDataset(X, y, name=MNIST)
        
    def load_letter(self) -> MLDataset:
        X, y = fetch_openml('letter', version=1, return_X_y=True, as_frame=False)
        return MLDataset(X, y, name=LETTER)
        
    def load_satimage(self) -> MLDataset:
        X, y = fetch_openml('satimage', version=1, return_X_y=True, as_frame=False)
        return MLDataset(X, y, name=SATIMAGE)
    
    def load_covtype(self, data_path="~/work/data/external") -> MLDataset:
        X,y = fetch_covtype(return_X_y=True, data_home=data_path)
        return MLDataset(X, y, name=COVTYPE)
    
    def load_wine(self) -> MLDataset:
        X,y = load_wine(return_X_y=True)
        return MLDataset(X,y,name=WINE)
                   
    def load_cars(self) -> MLDataset:
        # TODO: add ordinal encoding
        X, y = fetch_openml('car-evaluation', version=1, return_X_y=True, as_frame=False)
        return MLDataset(X, y, name=CARS)
                    
    def load_blood(self) -> MLDataset:
        X, y = fetch_openml('blood-transfusion-service-center', version=1, return_X_y=True, as_frame=False)
        return MLDataset(X, y, name=BLOOD)
                    
    def load_breast_cancer(self) -> MLDataset:
        X, y = load_breast_cancer(return_X_y=True)
        return MLDataset(X, y, name=BCANCER)
               
    def load_digits(self) -> MLDataset:
        X, y = load_digits(return_X_y=True) 
        return MLDataset(X, y, name=DIGITS)
              
    def load_vehicle(self) -> MLDataset: 
        X, y = fetch_openml('vehicle', version=1, return_X_y=True, as_frame=False) 
        return MLDataset(X, y, name=VEHICLE)