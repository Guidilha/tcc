import argparse
import numpy as np
import sklearn.metrics as mtr
import src.torchsetup as tsetup
from src.dataset.datasetcollector import DatasetCollector #load_dataset, DATASET_REGISTRIES
from src.models.hypernnclassifier import HyperNNClassifier
from src.models.primclassifier import PrimClassifier
from src.evaluation.modelevaluator import ModelEvaluator
from src.models.hypernnwithinitclassifier import HyperNNWithInitClassifier

def parse_experiments_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hypernn', type=str, help='which model to use: prim, hypernn.')
    parser.add_argument('--results_directory', default='results', type=str, help='where to store results.')
    parser.add_argument('--max_train_epochs', default=300, type=int, help='maximum number of epochs.')
    parser.add_argument('--max_number_of_boxes', default=1000, type=int, help='maximum number of boxes.')
    parser.add_argument('--patience', default=20, type=int, help='early stopping count.')
    parser.add_argument('--tau', default=3.0, type=float, help='initial smooth sigmoid factor.')
    parser.add_argument('--alpha', default=1.0, type=float, help='initial smoothmax factor.')
    parser.add_argument('--use_cuda', default='True', type=str, help='set device as cuda if available.')
    parser.add_argument('--init_strategy', default='som', type=str, choices=['som', 'random', 'kmeans', 'dbscan', 'xavier', 'all'], help='Estratégia de inicialização a ser testada isoladamente.')
    parser.add_argument('--prim_threshold', default=0.5, type=float, help='prim threshold for finding boxes.')
    parser.add_argument('--test_size', default=0.3, type=float, help='split ration for the test dataset.')
    parser.add_argument('--num_cv_folds', default=5, type=int, help='number of cv folds.')
    parser.add_argument('--num_seeds', default=3, type=int, help='number of random seeds.')
    parser.add_argument('--scaling', default=True, type=bool, help='apply feature standard scaling.')
    parser.add_argument('--use_small_data', default=True, type=bool, help='use only small datasets.')

    args, unknown = parser.parse_known_args()

    if isinstance(args.use_cuda, str): 
      args.use_cuda = args.use_cuda.lower() in ['true', '1', 'yes']
    return args

def get_hypernn_and_hyperparameters(experiment_options):
    device = tsetup.set_device(cuda=experiment_options.use_cuda)

    model = HyperNNWithInitClassifier()

    if experiment_options.init_strategy == 'all': 
      strategies = ['som', 'random', 'kmeans', 'dbscan'] 
    else: 
      strategies = [experiment_options.init_strategy]

    hparam_grid = {
        'init_strategy'           : [experiment_options.init_strategy],
        'som_sigma'               : [1.0],
        'som_lr'                  : [0.5],
        'gamma_escala'            : [1.0,8.0],
        #'dbscan_eps'              : [0.1,0.5,1],
        #'dbscan_min_samples'      : [1,5],
        'nboxes'                  : [2,6,12,20], #experiment_options.max_number_of_boxes],
        'device'                  : [device],
        'training_epochs'         : [experiment_options.max_train_epochs],
        'patience_early_stopping' : [experiment_options.patience],
        'learning_rate'           : [0.01, 0.05],
        'l1_reg'                  : [0], #1e-3
        'l2_reg'                  : [0], #1e-3
        'alpha_tau_decay_step'    : [10],
        'alpha'                   : [experiment_options.alpha],
        'tau'                     : [1.0,3.0, 5.0],
    }

    return model, hparam_grid

def main(experiment_options):
    model_name = experiment_options.model_name

    if model_name == 'prim':
        model, hyperparameters = get_prim_and_hyperparameters(experiment_options)
    elif model_name == 'hypernn':
        model, hyperparameters = get_hypernn_and_hyperparameters(experiment_options)
    else:
        return NotImplementedError(f'{model_name} is not implemented.')

    print(model_name)
    print(hyperparameters)

    #datasets = [ load_dataset(ds_name) for ds_name in DATASET_REGISTRIES["openml_datasets"] ]
    
    collector = DatasetCollector()
    
    if not experiment_options.use_small_data:
        datasets = collector.get_all_datasets()
    else:
        datasets = datasets = [
            collector.load_iris(),
            collector.load_wine(),
            collector.load_breast_cancer(),
            #collector.load_digits(),
            #collector.load_vehicle(),
        ]

    evaluator = ModelEvaluator(
        model = model,
        model_hyperparameter_space = hyperparameters,
        test_size=experiment_options.test_size,
        cross_validation_folds=experiment_options.num_cv_folds,
        scale_data=experiment_options.scaling,
        num_seeds=experiment_options.num_seeds,
        evaluation_metric=mtr.f1_score,
        results_logging_directory=experiment_options.results_directory,
    )

    evaluator.evaluate(datasets)

    print(evaluator.show_summary())

if __name__ == '__main__':
    main(parse_experiments_parameters())