import torch
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter
from os import path
from shared.checkpoint import CheckPoint
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy
from pruning.helpers import generate_random_mask_arr
from config import get_params, get_args_str
from preprocess import preprocess, use_sparsity_dict
from models.cnn_mnist import CNN_MNIST
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from gradient_estimators.coordinate_gradient_estimator import (
    CoordinateGradientEstimator as CGE,
)
from models.resnet import Resnet20
from models.cnn_fashion import CNN_FMNIST


def prepare_settings(args, device):
    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )

    elif args.dataset == "cifar10":
        model = Resnet20().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=args.momentum
        )

    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )

    if args.grad_estimate_method in ["rge-central", "rge-forward"]:
        method = args.grad_estimate_method[4:]
        print(f"Using RGE {method}")
        grad_estimator = RGE(
            model,
            mu=args.mu,
            num_pert=args.num_pert,
            grad_estimate_method=method,
            device=device,
        )
    elif args.grad_estimate_method in ["cge-forward"]:
        print("Using CGE forward")
        grad_estimator = CGE(
            model,
            mu=args.mu,
            device=device,
        )
    else:
        raise Exception(
            f"Grad estimate method {args.grad_estimate_method} not supported"
        )
    return model, criterion, optimizer, grad_estimator


def train_model(epoch: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    iter_per_epoch = len(train_loader)
    with tqdm(total=iter_per_epoch, desc="Training:") as t, torch.no_grad():
        for iteration, (images, labels) in enumerate(train_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            # update models
            optimizer.zero_grad()
            grad_estimator.compute_grad(images, labels, criterion)
            optimizer.step()

            pred = model(images)
            train_loss.update(criterion(pred, labels))
            train_accuracy.update(accuracy(pred, labels))
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)

    return train_loss.avg, train_accuracy.avg


def eval_model(epoch: int) -> tuple[float, float]:
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            pred = model(images)
            eval_loss.update(criterion(pred, labels))
            eval_accuracy.update(accuracy(pred, labels))
    print(
        f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg:.4f}, "
        f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


if __name__ == "__main__":
    args = get_params().parse_args()
    args.dataset = "cifar10"
    torch.manual_seed(args.seed)

    device, train_loader, test_loader = preprocess(args)
    model, criterion, optimizer, grad_estimator = prepare_settings(args, device)

    checkpoint = CheckPoint(args, model, optimizer, grad_estimator)
    checkpoint_trained_epochs = checkpoint.get_trained_epochs()

    args_str = get_args_str(args) + "-" + model.model_name
    if args.log_to_tensorboard:
        tensorboard_sub_folder = args_str + "-" + get_current_datetime_str()
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    sparsity_dict = use_sparsity_dict(args, model.model_name)
    for epoch in range(args.epoch):
        if sparsity_dict is not None and epoch % args.mask_shuffle_interval == 0:
            print("Updating gradient mask!")
            mask_arr = generate_random_mask_arr(model, sparsity_dict, device)
            grad_estimator.set_prune_mask_arr(mask_arr)

        train_loss, train_accuracy = train_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar(
                "Loss/train", train_loss, checkpoint_trained_epochs + epoch
            )
            writer.add_scalar(
                "Accuracy/train", train_accuracy, checkpoint_trained_epochs + epoch
            )
        eval_loss, eval_accuracy = eval_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/test", eval_loss, checkpoint_trained_epochs + epoch)
            writer.add_scalar(
                "Accuracy/test", eval_accuracy, checkpoint_trained_epochs + epoch
            )

        if checkpoint.should_update(eval_loss, eval_accuracy, epoch):
            file_name = (
                model.model_name
                + f"-epoch-{checkpoint_trained_epochs + epoch + 1}"
                + f"-acc-{eval_accuracy * 100:.2f}"
            )
            checkpoint.save(
                file_name,
                epoch,
                subfolder=args.log_to_tensorboard,
            )

    if args.log_to_tensorboard:
        writer.close()
