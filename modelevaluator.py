import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from tqdm.auto import trange
import sklearn.metrics as mtr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from ..torchsetup import set_seeds
from ..utils import binarize_labels
from .hyperparametersearch import HyperparameterGridSearch
from src.evaluation.tensorboard_logger import HyperNNTensorBoardLogger

class ModelEvaluator:
    def __init__(self, model, model_hyperparameter_space, test_size, cross_validation_folds, scale_data, num_seeds, evaluation_metric, results_logging_directory, experiment_name=None, random_state=42):
        self.model = model
        self.cross_validation_folds = cross_validation_folds
        self.evaluation_metric = evaluation_metric
        self.test_size = test_size
        self.num_seeds = num_seeds
        self.model_hyperparameter_space = model_hyperparameter_space
        self.scale_data = scale_data
        self.results = []
        self.logger = None
        self.results_filepath = ''
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.configure_logging_and_results(results_logging_directory)
        
    def configure_logging_and_results(self, results_logging_directory):
        directory_path = Path(results_logging_directory)
        assert directory_path.exists(), f'The provided path {results_logging_directory} does not exist.'
        
        model_name = type(self.model).__name__
        datetime_str = time.strftime('%Y%m%d-%H%M%S')
        exp_name = ''
        
        if not self.experiment_name is None:
            exp_name = f'_{self.experiment_name}'
        
        filename = f'results_{model_name}_{datetime_str}{exp_name}'
        
        logging_filepath = directory_path.joinpath(f'{filename}.log')
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(filename=logging_filepath),
            ]
        )
    
        self.logger = logging.getLogger("logger")
        self.results_filepath = directory_path.joinpath(f'{filename}.csv')
        
    def create_experiment_report(self, dataset_name, dataset_label, best_model_hyperparameters, training_scores, test_scores, hyperparameter_search_time, training_time,initialization_time, prediction_time, final_nboxes, seed):
        '''Creates a dictionary with all computed metrics, results, and experiment information.'''
        report = {
            'model': type(self.model).__name__,
            'hyperparameter_space': self.model_hyperparameter_space,
            'dataset': dataset_name,
            'dataset_label': dataset_label,
            'scale_data': self.scale_data,
            'evaluation_metric': self.evaluation_metric.__name__,
            'cross_validation_folds': self.cross_validation_folds,
            'test_size': self.test_size,
            'finalnboxes': final_nboxes,
            **best_model_hyperparameters,
            **training_scores,
            **test_scores,
            'training_time': training_time,
            'initialization_time': initialization_time,
            'hyperparameter_search_time': hyperparameter_search_time,
            'prediction_time': prediction_time,
            'seed': seed,
        }
        
        return report
    
    def compute_metric_scores(self, y_true, y_pred, prefix=''):
        '''Computes evaluation metrics'''
        
        scores = {
            f'{prefix}accuracy': mtr.accuracy_score(y_true, y_pred),
            f'{prefix}precision': mtr.precision_score(y_true, y_pred),
            f'{prefix}recall': mtr.recall_score(y_true, y_pred),
            f'{prefix}f1_score': mtr.f1_score(y_true, y_pred),
            f'{prefix}balanced_accuracy': mtr.balanced_accuracy_score(y_true, y_pred),
        }
        
        return scores
    
    def evaluate_model_on_dataset(self, X, y, dataset_name, seed):
        '''Evaluates model using StratifiedKFold cross-validation.'''
        results = []
        distinct_labels = np.unique(y)
        
        # Evaluate model using one-vs-all
        for i, label in enumerate(distinct_labels):
            self.logger.info(f'Seed: {seed}, Dataset: {dataset_name}, Label: {label}, {i}/{len(distinct_labels)}.')
            
            # Adjust current labels
            y_current = binarize_labels(y, label)
            
            # Split train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y_current, test_size=self.test_size, stratify=y_current, random_state=seed)
            
            if self.scale_data:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            
            gridsearch = HyperparameterGridSearch(
                self.model, 
                self.model_hyperparameter_space, 
                num_cross_validation_folds=self.cross_validation_folds,
                evaluation_metric=self.evaluation_metric, 
                random_state=seed)
            
            # Search for hyperparameters
            start_grid_search_time = time.time()
            best_model = gridsearch.fit(X_train, y_train)
            end_grid_search_time = time.time()
            
            grid_search_time = end_grid_search_time - start_grid_search_time
            tb_logger = HyperNNTensorBoardLogger(log_dir="runs")
            # Train best model
            start_train_time = time.time()
            best_model.fit(X_train, y_train)
            end_train_time = time.time()
            
            training_time = end_train_time - start_train_time
            
            # Compute training scores
            y_pred_train = best_model.predict(X_train)
            train_scores = self.compute_metric_scores(y_train, y_pred_train, prefix='train_')
            
            # Predict test data using best model
            start_prediction_time = time.time()
            y_pred_test = best_model.predict(X_test)
            end_prediction_time = time.time()
            
            prediction_time = end_prediction_time - start_prediction_time
            
            # Compute test scores
            test_scores = self.compute_metric_scores(y_test, y_pred_test, prefix='test_')
            initialization_time = getattr(best_model, 'init_time', 0.0)
            # Create results report
            curret_result_report = self.create_experiment_report(
                dataset_name=dataset_name,
                dataset_label=label,
                best_model_hyperparameters=gridsearch.best_model_hyperparameters, 
                training_scores=train_scores, 
                test_scores=test_scores, 
                hyperparameter_search_time=grid_search_time, 
                training_time=training_time, 
                initialization_time=initialization_time,
                prediction_time=prediction_time, 
                final_nboxes = best_model.get_num_boxes(),
                seed=seed)
            result_entry = { 
                "dataset": dataset_name, 
                "test_f1_score": test_scores, 
                "training_time": training_time, 
                "init_time": initialization_time, 
                "prediction_time": prediction_time, 
                "final_nboxes": best_model.get_num_boxes(), 
                "best_params": gridsearch.best_model_hyperparameters, } 
            self.results.append(result_entry)
             
            tb_logger = HyperNNTensorBoardLogger(log_dir="runs") 
            tb_logger.log_experiment_result( 
                dataset_name=dataset_name, 
                strategy_name=getattr(best_model, 'init_strategy', 'default'), 
                model=best_model, 
                metrics_dict={ 
                    **test_scores,
                    "training_time": training_time, 
                    "initialization_time": initialization_time, 
                    "prediction_time": prediction_time, 
                    "final_nboxes": result_entry["final_nboxes"] 
                    }, 
                seed=seed
            )
                
            
            # Log current results
            self.logger.info(curret_result_report)
            
            # Store current results
            results.append(curret_result_report)
            
        return results
            
    def evaluate(self, datasets):
        '''Evaluates a given model over a list of MLDatasets.'''
        
        # Set initial seed for generating further random seeds
        #np.random.seed(self.random_state)
        #np.random.randint(0, 1234, size=self.num_seeds)
        
        num_experiments = self.num_seeds * len(datasets)
        self.logger.info(f'Starting {num_experiments} experiments.')
        
        self.results = []
        
        with trange(num_experiments) as exp_pbar:
            exp_pbar.set_description('Running experiments')
            for dt in datasets:
                for seed in range(self.num_seeds):
                    set_seeds(seed)
                    exp_pbar.set_postfix(dataset=dt.name, seed=seed)
                    
                    results = self.evaluate_model_on_dataset(
                        X = dt.X,
                        y = dt.y,
                        dataset_name = dt.name,
                        seed = seed
                    ) 
                    
                    self.results.extend(results)
                    exp_pbar.update(1)
            
        self.logger.info('Final results:')
        self.logger.info(self.results)
        self.save_results()

    def save_results(self):
        '''Saves experiment results in the specified directory and 
        returns results as a pandas.DataFrame.'''
        
        experiment_results = pd.DataFrame(self.results)
        experiment_results.to_csv(self.results_filepath)
        
        return experiment_results 
    
    def show_summary(self):
        '''Shows summary of results in terms of Test F1-Score, 
        Training time, Prediction time, and Hyperparameter-search time.'''
        
        experiment_results = pd.DataFrame(self.results)
        
        df_pivot = experiment_results.pivot_table(
            columns=['model', 'dataset'], 
            values=['finalnboxes',
                    'test_f1_score',
                    'training_time', 
                    'prediction_time', 
                    'hyperparameter_search_time',
                    'initialization_time'])
        
        return df_pivot