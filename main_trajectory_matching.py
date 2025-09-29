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
from Data.data_trajectory_matching import get_matching_datasets, get_n_classes, pick_poisoner, load_dataset
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

    config['num_classes'] = get_n_classes(config['dataset'])

    poisoner = pick_poisoner(config['poisoner'],
                             config['dataset'],
                             config['target_label'])
    
    poison_train, _, test, poison_test, _ =\
        get_matching_datasets(config['dataset'], poisoner, config['source_label'], train_pct=config['train_pct'])

    clean_train = load_dataset(config['dataset'], train=True)
    clean_test = load_dataset(config['dataset'], train=False)

    poisoned_loader = torch.utils.data.DataLoader(poison_train, batch_size=config['batch_size'], shuffle=True)
    train_loader = torch.utils.data.DataLoader(clean_train, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(clean_test, batch_size=config['batch_size'], shuffle=False)

    logging.info(f"Train loader size: {len(clean_train)}")
    logging.info(f"Test loader size: {len(clean_test)}")
    logging.info(f"Poison data loader size: {len(poison_train)}")

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
        model, expert_model, _loader, poisoned_loader, criterion,
        budget=math.ceil(config['batch_size'] * config['budget_ratio']), 
        controlled_subset_size=config['controlled_subset_size'], 
        steps=config['byzantine_steps'], 
        lr=config['byzantine_lr'],
        random_restart=config['random_restart']
    ) for _loader in byzantine_loaders]

    workers = honest_list + byzantine_list
    aggregator = Aggregator(model, workers, config['aggregation_method'])
    aggregator.train(test_loader, config['source_label'], epochs=config['epochs'])

if __name__ == "__main__":
    main()
