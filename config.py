import torch

config = {
    'seed': 42,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'log_file': 'logs/training.log',
    
    'train_size': 10000,
    'test_size': 1000,
    'targeted_data_size': 1000,

    'batch_size': 64,
    'test_batch_size': 1000,
    'num_classes': 10,

    'dataset': 'cifar',
    'train_pct': 1.0,  # percentage of the training set to use
    'poisoner': '1xp', # '1xp', '2xp', '3xp', '1xs', '2xs', '1xl', '4xl'
    
    'model_type': 'CNN',  # 'MLP' or 'LogisticRegression'
    
    'epochs': 200,
    'rounds_per_epoch': 200,
    
    'random_restart': 1,  # number of random initializations for the byzantine attack
    'optimizing_method': 'logits',  # 'logits' 
    'flip_strategies': ["sim_variation"],
    'budget_ratio': 0.25, #0.25
    'controlled_subset_size': 1.0, #1.0
    'byzantine_steps': 250,
    'byzantine_lr': 0.01,
    'target_label': 5,
    'source_label': 3,

    'aggregation_method': 'mean',  # 'mean' or 'median'
    'num_honest_workers': 0,
    'num_byzantine_workers': 1,
}