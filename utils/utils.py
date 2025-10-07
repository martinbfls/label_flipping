import logging
import random
import os
import math
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from Models.models import MLP, LogisticRegression, CNN
from Models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
import argparse
from types import SimpleNamespace


to_tensor = ToTensor()

def dict_to_namespace(d):
    return argparse.Namespace(**d)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_round(args):
    num_sample_per_worker = math.ceil(args.train_size / (args.num_honest_workers + args.num_byzantine_workers))
    num_rounds_per_epoch = math.ceil(num_sample_per_worker / args.batch_size)
    args.rounds_per_epoch = num_rounds_per_epoch

def collate_fn_mnist(batch):
    return batch

def collate_fn(batch):
    imgs, targets = zip(*batch)
    processed_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            processed_imgs.append(img)
        else:
            processed_imgs.append(to_tensor(img))
    imgs = torch.stack(processed_imgs)
    targets = torch.tensor(targets)
    return imgs, targets

def select_collate_fn(dataset):
    if dataset.lower() in ['mnist']:
          return collate_fn_mnist
    else:
        return collate_fn

def load_model(model_type, input_shape, num_classes):
    if model_type == 'MLP':
        return MLP(input_shape=input_shape, hidden_size=512, num_classes=num_classes)
    elif model_type == 'CNN':
        return CNN(input_shape=input_shape, num_classes=num_classes)
    elif model_type == 'ResNet20':
        return resnet20(num_classes=num_classes)
    elif model_type == 'ResNet32':
        return resnet32(num_classes=num_classes)
    elif model_type == 'ResNet44':
        return resnet44(num_classes=num_classes)
    elif model_type == 'ResNet56':
        return resnet56(num_classes=num_classes)
    elif model_type == 'ResNet110':
        return resnet110(num_classes=num_classes)
    elif model_type == 'ResNet1202':
        return resnet1202(num_classes=num_classes)
    else:
        return LogisticRegression(input_shape=input_shape, num_classes=num_classes)

def setup_save_path(args):
    output_dir = f"results/{args.attack_method}/{args.loss_type}/{args.dataset}/{args.model_type}/agg-{args.aggregation_method}_honest-{args.num_honest_workers}_byzantine-{args.num_byzantine_workers}_controlled-{args.controlled_subset_size}_budget-{args.budget_ratio}_steps-{args.byzantine_steps}_lr-{args.byzantine_lr}_adv_sch_{args.adversarial_scheduler}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return f"{output_dir}.png"

def log_file(args):
    output_dir = f"logs/{args.attack_method}/{args.loss_type}/{args.dataset}/{args.model_type}/agg-{args.aggregation_method}_honest-{args.num_honest_workers}_byzantine-{args.num_byzantine_workers}_controlled-{args.controlled_subset_size}_budget-{args.budget_ratio}_steps-{args.byzantine_steps}_lr-{args.byzantine_lr}_adv_sch_{args.adversarial_scheduler}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return f"{output_dir}output.log"

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def setup_optimizer(model, optim_type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0):
    if optim_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type {optim_type}")
    
def setup_scheduler(optimizer, sched_type='StepLR', step_size=10, gamma=0.1):
    if sched_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=gamma)
    elif sched_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    elif sched_type == 'None':
        return None
    else:
        raise ValueError(f"Unknown scheduler type {sched_type}")

