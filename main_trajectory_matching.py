import torch
import torch.nn as nn
import random
import math
import copy
import logging
from models import MLP, LogisticRegression
from Worker import Worker
from ByzantineWorkers.ByzantineWorkerGlobalTrajectoryMatching import ByzantineWorkerGlobalTrajectoryMatching
from Aggregator import Aggregator
from data import load_mnist_trajectory_matching
from config import config

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_round():
    num_sample_per_worker = math.ceil(config['train_size'] / (config['num_honest_workers'] + config['num_byzantine_workers']))
    num_rounds_per_epoch = math.ceil(num_sample_per_worker / config['batch_size'])
    config['rounds_per_epoch'] = num_rounds_per_epoch

def main():
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=config['log_file'],
        filemode='w'
    )
    
    logging.info(f"Using device: {config['device']}")
    logging.info(f"Configuration: {config}")

    set_seed(config['seed'])
    set_round()

    train_loader, test_loader, targeted_data_loader = load_mnist_trajectory_matching(train_size=config['train_size'], 
                                                                 test_size=config['test_size'], 
                                                                 batch_size=config['batch_size'], 
                                                                 test_batch_size=config['test_batch_size'], 
                                                                 target_label=config['target_label'], 
                                                                 adversarial_label=config['adversarial_label'])

    logging.info(f"Train loader size: {len(train_loader.dataset)}")
    logging.info(f"Test loader size: {len(test_loader.dataset)}")
    logging.info(f"Targeted data loader size: {len(targeted_data_loader.dataset)}")

    if config['model_type'] == 'MLP':
        model = MLP().to(config['device'])
    else:
        model = LogisticRegression().to(config['device'])
    
    expert_model = copy.deepcopy(model)

    criterion = nn.CrossEntropyLoss()

    num_workers = config['num_honest_workers'] + config['num_byzantine_workers']
    
    if num_workers == 0:
        raise ValueError("Total number of workers must be greater than zero.")
    
    dataset_size = len(train_loader.dataset)
    base_size = dataset_size // num_workers
    remainder = dataset_size % num_workers
    
    worker_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_workers)]
    
    worker_loaders = torch.utils.data.random_split(train_loader.dataset, worker_sizes)
    worker_loaders = [torch.utils.data.DataLoader(subset, batch_size=config['batch_size'], shuffle=True) for subset in worker_loaders]

    byzantine_loaders = worker_loaders[:config['num_byzantine_workers']]
    honest_loaders = worker_loaders[config['num_byzantine_workers']:]

    honest_list = [Worker(model, _loader, criterion) for _loader in honest_loaders]
    byzantine_list = [ByzantineWorkerGlobalTrajectoryMatching(
        model, expert_model, _loader, targeted_data_loader, criterion,
        budget=math.ceil(config['batch_size'] * config['budget_ratio']), 
        controlled_subset_size=config['controlled_subset_size'], 
        steps=config['byzantine_steps'], 
        lr=config['byzantine_lr'],
        random_restart=config['random_restart']
    ) for _loader in byzantine_loaders]

    workers = honest_list + byzantine_list
    aggregator = Aggregator(model, workers, config['aggregation_method'])
    aggregator.train(test_loader, config['target_label'], epochs=config['epochs'])

if __name__ == "__main__":
    main()
