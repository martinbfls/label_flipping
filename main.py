import torch, math, copy, logging, argparse
import torch.nn as nn
from utils.utils import (
    dict_to_namespace, set_seed, set_round, select_collate_fn, load_model,
    setup_logger, setup_optimizer, setup_scheduler, log_file, setup_save_path
)
from Worker import Worker
from ByzantineWorkers.ByzantineWorkerGlobalTrajectoryMatching import ByzantineWorkerGlobalTrajectoryMatching
from ByzantineWorkers.ByzantineWorkerLocalTrajectoryMatching import ByzantineWorkerLocalTrajectoryMatching
from ByzantineWorkers.ByzantineWorkerWitch import ByzantineWorkerWitch
from Aggregator import Aggregator
from Data.data_trajectory_matching import (
    get_matching_datasets, get_n_classes, pick_poisoner, limit_dataset
)
from Data.data import load_mnist
from utils.showing_results import plot_datasets_differences

def setup_experiment(args):
    if isinstance(args, dict):
        args = dict_to_namespace(args)
    setup_logger(log_file(args))
    logging.info(f"Using device: {args.device}")
    logging.info(f"Arguments: {args}")
    set_seed(); set_round(args)
    return args

def build_model(args, input_shape):
    return load_model(args.model_type, input_shape, get_n_classes(args.dataset.lower())).to(args.device)

def split_workers(model, dataset, save_path, args, criterion, **byz_kwargs):
    n_workers = args.num_honest_workers + args.num_byzantine_workers
    if n_workers == 0:
        raise ValueError("Total number of workers must be > 0.")
    base, rem = divmod(len(dataset), n_workers)
    sizes = [base + 1 if i < rem else base for i in range(n_workers)]
    collate_fn = select_collate_fn(args.model_type)
    splits = torch.utils.data.random_split(dataset, sizes)
    loaders = [torch.utils.data.DataLoader(s, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for s in splits]
    honest, byz = loaders[args.num_byzantine_workers:], loaders[:args.num_byzantine_workers]
    honest_workers = [Worker(model, l, criterion, id=i) for i, l in enumerate(honest)]
    byz_workers = [byz_kwargs["cls"](model=model, loader=l, criterion=criterion, id=i, save_path=save_path, **byz_kwargs["params"]) for i, l in enumerate(byz)]
    return honest_workers + byz_workers

def build_aggregator(model, workers, save_path, args):
    opt = setup_optimizer(model, args.aggregator_optim, lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched = setup_scheduler(opt, sched_type=args.aggregator_scheduler, step_size=20, gamma=0.1) if args.aggregator_scheduler else None
    return Aggregator(model, workers, opt, sched, save_path, args.aggregation_method)

def prepare_datasets(args):
    poisoner = pick_poisoner(args.poisoner, args.dataset, args.target_label)
    clean_train, poison_train, _, clean_test, poisoned_test, _ = get_matching_datasets(dataset_flag=args.dataset, 
                                                                                       poisoner=poisoner, 
                                                                                       label=args.source_label, 
                                                                                       train_pct=args.train_pct)
    clean_train, clean_test = limit_dataset(clean_train, 7000), limit_dataset(clean_test, 5000)
    poison_train, poisoned_test = limit_dataset(poison_train, 7000), limit_dataset(poisoned_test, 5000)

    collate = lambda shuffle: dict(shuffle=shuffle, batch_size=args.batch_size, collate_fn=select_collate_fn(args.model_type))
    poisoned_loader = torch.utils.data.DataLoader(poison_train, **collate(True))
    poisoned_test_loader = torch.utils.data.DataLoader(poisoned_test, **collate(False))
    train_loader = torch.utils.data.DataLoader(clean_train, **collate(True))
    test_loader = torch.utils.data.DataLoader(clean_test, **collate(False))

    args.train_size, args.test_size, args.targeted_data_size = map(len, [train_loader.dataset, test_loader.dataset, poisoned_loader.dataset])
    set_round(args)
    logging.info(f"Sizes — Poison train: {len(poison_train)}, Clean train: {len(clean_train)}, Clean test: {len(clean_test)}")
    return train_loader, test_loader, poisoned_loader, poisoned_test_loader

def main(args):
    args = setup_experiment(args)
    save_path = setup_save_path(args)
    if args.attack_method == "witch":
        train_loader, test_loader, targeted_loader = load_mnist(
            train_size=args.train_size, test_size=args.test_size,
            batch_size=args.batch_size, test_batch_size=args.batch_size,
            target_label=args.source_label, targeted_data_size=args.targeted_data_size
        )
        sample_batch = next(iter(train_loader))[0]
        model = build_model(args, tuple(sample_batch.shape[1:]))
        criterion = nn.CrossEntropyLoss()
        workers = split_workers(model, train_loader.dataset, save_path, args, criterion,
            cls=ByzantineWorkerWitch,
            params=dict(
                target_loader=targeted_loader, source_label=args.source_label,
                target_label=args.target_label, scheduler=args.adversarial_scheduler,
                budget=math.ceil(args.batch_size * args.budget_ratio),
                controlled_subset_size=args.controlled_subset_size,
                steps=args.byzantine_steps, lr=args.byzantine_lr
            )
        )
        results = build_aggregator(model, workers, save_path, args).train(test_loader, test_loader, args.source_label, args.target_label, 
                                                                epochs=args.epochs, round_per_epoch=args.rounds_per_epoch)
        return results

    elif args.attack_method in ["global_trajectory_matching", "local_trajectory_matching"]:
        train_loader, test_loader, poisoned_loader, poisoned_test_loader = prepare_datasets(args)
        plot_datasets_differences(train_loader.dataset, poisoned_loader.dataset, save_path.replace('.png', 'train.png'), args.source_label, args.target_label, inputs_or_labels='both', n_samples=5)
        plot_datasets_differences(test_loader.dataset, poisoned_test_loader.dataset, save_path.replace('.png', 'test.png'), args.source_label, args.target_label, inputs_or_labels='both', n_samples=5)
        sample_batch = next(iter(train_loader))[0]
        model = build_model(args, tuple(sample_batch[0].shape))
        criterion = nn.CrossEntropyLoss()
        cls = ByzantineWorkerGlobalTrajectoryMatching if args.attack_method == "global_trajectory_matching" else ByzantineWorkerLocalTrajectoryMatching
        params = dict(
            poisoned_loader=poisoned_loader, scheduler=args.adversarial_scheduler,
            budget=math.ceil(args.batch_size * args.budget_ratio),
            controlled_subset_size=args.controlled_subset_size,
            steps=args.byzantine_steps, lr=args.byzantine_lr,
            random_restart=args.random_restarts, loss_type=args.loss_type
        )
        if args.attack_method == "global_trajectory_matching":
            params["expert_model"] = copy.deepcopy(model)
        workers = split_workers(model, train_loader.dataset, save_path, args, criterion, cls=cls, params=params)
        results = build_aggregator(model, workers, save_path, args).train(test_loader, poisoned_test_loader, args.source_label, args.target_label,
                                                                epochs=args.epochs, round_per_epoch=args.rounds_per_epoch)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_method", type=str, required=True, choices=["witch", "global_trajectory_matching", "local_trajectory_matching"])
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
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "cosine_similarity"])
    parser.add_argument("--source_label", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--poisoner", type=str, default="simple")
    parser.add_argument("--train_pct", type=float, default=1.0)
    parser.add_argument("--rounds_per_epoch", type=int, default=10)

    args = parser.parse_args()
    main(args)