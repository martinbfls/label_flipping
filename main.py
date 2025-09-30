import torch
import torch.nn as nn
import math, copy, logging, argparse
from utils.utils import dict_to_namespace, set_seed, set_round, select_collate_fn, load_model, \
                        setup_logger, setup_optimizer, setup_scheduler, log_file
from Worker import Worker
from ByzantineWorkers.ByzantineWorkerGlobalTrajectoryMatching import ByzantineWorkerGlobalTrajectoryMatching
from ByzantineWorkers.ByzantineWorkerWitch import ByzantineWorkerWitch
from Aggregator import Aggregator
from Data.data_trajectory_matching import get_matching_datasets, get_n_classes, pick_poisoner, load_dataset
from Data.data import load_mnist

def setup_experiment(args):
    if isinstance(args, dict):
        args = dict_to_namespace(args)
    setup_logger(log_file(args))
    logging.info(f"Using device: {args.device}")
    logging.info(f"Arguments: {args}")
    set_seed()
    set_round(args)
    return args

def build_model(args, input_shape):
    num_classes = get_n_classes(args.dataset.lower())
    return load_model(args.model_type, input_shape, num_classes).to(args.device)

def split_workers(model, train_dataset, args, criterion, **byz_kwargs):
    num_workers = args.num_honest_workers + args.num_byzantine_workers
    if num_workers == 0:
        raise ValueError("Total number of workers must be > 0.")
    dataset_size = len(train_dataset)
    base_size, remainder = dataset_size // num_workers, dataset_size % num_workers
    worker_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_workers)]
    collate_fn = select_collate_fn(args.model_type)
    worker_loaders = torch.utils.data.random_split(train_dataset, worker_sizes)
    worker_loaders = [torch.utils.data.DataLoader(s, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for s in worker_loaders]
    byz_loaders = worker_loaders[:args.num_byzantine_workers]
    honest_loaders = worker_loaders[args.num_byzantine_workers:]
    honest_list = [Worker(model, l, criterion) for l in honest_loaders]
    byz_list = [byz_kwargs["cls"](model=model, loader=l, criterion=criterion, **byz_kwargs["params"]) for l in byz_loaders]
    return honest_list + byz_list

def build_aggregator(model, workers, args):
    opt = setup_optimizer(model, args.aggregator_optim, lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched = setup_scheduler(opt, sched_type=args.aggregator_scheduler, step_size=20, gamma=0.1) if args.aggregator_scheduler else None
    return Aggregator(model, workers, opt, sched, args.aggregation_method)

def main(args):
    args = setup_experiment(args)

    if args.attack_method == "witch":
        train_loader, test_loader, targeted_loader = load_mnist(
            train_size=args.train_size, test_size=args.test_size,
            batch_size=args.batch_size, test_batch_size=args.test_batch_size,
            target_label=args.source_label, targeted_data_size=args.targeted_data_size
        )
        sample_batch = next(iter(train_loader))[0]
        model = build_model(args, tuple(sample_batch.shape[1:]))
        criterion = nn.CrossEntropyLoss()
        workers = split_workers(model, train_loader.dataset, args, criterion,
            cls=ByzantineWorkerWitch,
            params=dict(targeted_data=targeted_loader, target_label=args.source_label,
                        adversarial_label=args.target_label, scheduler=args.adversarial_scheduler,
                        budget=math.ceil(args.batch_size * args.budget_ratio),
                        controlled_subset_size=args.controlled_subset_size, steps=args.byzantine_steps,
                        lr=args.byzantine_lr)
        )
        aggregator = build_aggregator(model, workers, args)
        aggregator.train(test_loader, args.source_label, epochs=args.epochs, round_per_epoch=args.rounds_per_epoch)

    elif args.attack_method == "trajectory_matching":
        poisoner = pick_poisoner(args.poisoner, args.dataset, args.target_label)
        poison_train, _, _, _, _ = get_matching_datasets(args.dataset, poisoner, args.source_label, train_pct=args.train_pct)
        clean_train = load_dataset(args.dataset, train=True)
        clean_test = load_dataset(args.dataset, train=False)
        poisoned_loader = torch.utils.data.DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, collate_fn=select_collate_fn(args.model_type))
        train_loader = torch.utils.data.DataLoader(clean_train, batch_size=args.batch_size, shuffle=True, collate_fn=select_collate_fn(args.model_type))
        test_loader = torch.utils.data.DataLoader(clean_test, batch_size=args.batch_size, shuffle=False, collate_fn=select_collate_fn(args.model_type))
        args.train_size = len(train_loader.dataset)
        args.test_size = len(test_loader.dataset)
        args.targeted_data_size = len(poisoned_loader.dataset)
        set_round(args)
        sample_batch = next(iter(train_loader))[0]
        model = build_model(args, tuple(sample_batch[0].shape))
        expert_model = copy.deepcopy(model)
        criterion = nn.CrossEntropyLoss()
        workers = split_workers(model, train_loader.dataset, args, criterion,
            cls=ByzantineWorkerGlobalTrajectoryMatching,
            params=dict(expert_model=expert_model, poisoned_loader=poisoned_loader,
                        scheduler=args.adversarial_scheduler,
                        budget=math.ceil(args.batch_size * args.budget_ratio),
                        controlled_subset_size=args.controlled_subset_size,
                        steps=args.byzantine_steps, lr=args.byzantine_lr,
                        random_restart=args.random_restarts)
        )
        aggregator = build_aggregator(model, workers, args)
        aggregator.train(test_loader, args.source_label, epochs=args.epochs, round_per_epoch=args.rounds_per_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_method", type=str, required=True, choices=["witch", "trajectory_matching"])
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model_type", type=str, default="CNN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_honest_workers", type=int, default=5)
    parser.add_argument("--num_byzantine_workers", type=int, default=1)
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--controlled_subset_size", type=int, default=32)
    parser.add_argument("--byzantine_steps", type=int, default=1)
    parser.add_argument("--byzantine_lr", type=float, default=0.01)
    parser.add_argument("--random_restart", action="store_true")
    parser.add_argument("--aggregation_method", type=str, default="average")
    parser.add_argument("--aggregator_optim", type=str, default="sgd")
    parser.add_argument("--aggregator_scheduler", type=str, default=None)
    parser.add_argument("--adversarial_scheduler", type=str, default=None)
    parser.add_argument("--source_label", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--poisoner", type=str, default="simple")
    parser.add_argument("--train_pct", type=float, default=1.0)
    parser.add_argument("--rounds_per_epoch", type=int, default=10)

    args = parser.parse_args()
    main(args)