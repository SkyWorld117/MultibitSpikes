from spikingjelly.activation_based.model.train_classify import Trainer


import datetime
import os
import time
import warnings
from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
import torch
import torch.utils.data
import torchvision
from spikingjelly.activation_based.model.tv_ref_classify.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.model.train_classify import set_deterministic
from torchvision import datasets
import tonic
try:
    from torchvision import prototype
except ImportError:
    prototype = None

from surrogate import Sigmoid
from neurons import LIFNode
import models

class RandomTrainer(Trainer):
    def __init__(self, N=1, T=10, args=None):
        super().__init__()
        self.model = None
        self.train_nnz = None
        self.test_nnz = None

        self.args = args

        self.train = False

        self.N = N
        self.T = T

        self.train_acc1 = []
        self.test_acc1 = []
        if args is not None and args.quantize:
            self.quant_acc1 = 0.0

        self.reached_target_train_acc = False
        self.train_iters = 0
        self.reached_target_test_acc = False
        self.test_iters = 0

    def cal_acc1_acc5(self, output, target):
        # define how to calculate acc1 and acc5
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        if self.train:
            self.train_acc1.append(acc1.item() / 100.0)
            if acc1.item() / 100.0 >= self.args.acc:
                self.reached_target_train_acc = True
            if not self.reached_target_train_acc:
                self.train_iters += 1

        return acc1, acc5
    
    def train_one_epoch(self, model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
        self.train = True

        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, -1, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                image = self.preprocess_train_sample(args, image)

                out_fr = 0.
                for i in range(self.T):
                    out_fr += model(image[i])
                for i in range(model.num_spiking + 1):
                    self.train_nnz[i].append(np.mean(model.nnz[i]))
                model.nnz = [ [] for _ in range(model.num_spiking + 1) ]

                output = self.process_model_output(args, out_fr)

                loss = criterion(output, target)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            functional.reset_net(model)

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = self.cal_acc1_acc5(output, target)
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')
        return train_loss, train_acc1, train_acc5
    
    def evaluate(self, args, model, criterion, data_loader, device, log_suffix=""):
        self.train = False
        
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)

                out_fr = 0.
                for i in range(self.T):
                    out_fr += model(image[i])
                for i in range(model.num_spiking + 1):
                    self.test_nnz[i].append(np.mean(model.nnz[i]))
                model.nnz = [ [] for _ in range(model.num_spiking + 1) ]

                output = self.process_model_output(args, out_fr)
                
                loss = criterion(output, target)

                acc1, acc5 = self.cal_acc1_acc5(output, target)
                batch_size = target.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
                functional.reset_net(model)
        # gather the stats from all processes

        num_processed_samples = utils.reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes()

        test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')

        self.test_acc1.append(test_acc1 / 100.0)
        if not self.reached_target_test_acc:
            self.test_iters += 1
        if test_acc1 / 100.0 >= self.args.acc:
            self.reached_target_test_acc = True
        return test_loss, test_acc1, test_acc5
    
    def evaluate_quant(self, args, model, data_loader):
        self.train = False

        model.to('cpu')
        torch.backends.quantized.engine = 'qnnpack'
        torch.quantization.convert(model, inplace=True)
        model.eval()

        with torch.inference_mode():
            for image, target in data_loader:
                image = image.to('cpu', non_blocking=True)
                target = target.to('cpu', non_blocking=True)
                image = self.preprocess_test_sample(args, image)

                out_fr = 0.
                for i in range(self.T):
                    out_fr += model(image[i])

                output = self.process_model_output(args, out_fr)
                acc1, acc5 = self.cal_acc1_acc5(output, target)
                self.quant_acc1 += acc1.item() * target.shape[0]

                functional.reset_net(model)
        
        self.quant_acc1 /= len(data_loader.dataset) * 100.0
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        if args.dataset in ['CIFAR10', 'FashionMNIST', 'MNIST']:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # [N, C, H, W] -> [T, N, C, H, W]
        if args.bf16:
            x = x.bfloat16()
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        if args.dataset in ['CIFAR10', 'FashionMNIST', 'MNIST']:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # [N, C, H, W] -> [T, N, C, H, W]
        if args.bf16:
            x = x.bfloat16()
        return x

    def process_model_output(self, args, y: torch.Tensor):
        # return y.sum(0)  # return firing rate
        return y

    def get_args_parser(self, add_help=True):

        parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

        parser.add_argument("--data-path", default=None, type=str, help="dataset path")
        parser.add_argument("--model", default=None, type=str, help="model name")
        parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
        parser.add_argument(
            "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
        )
        parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
        parser.add_argument(
            "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
        )
        parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
        parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=0.,
            type=float,
            metavar="W",
            help="weight decay (default: 0.)",
            dest="weight_decay",
        )
        parser.add_argument(
            "--norm-weight-decay",
            default=None,
            type=float,
            help="weight decay for Normalization layers (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)", dest="label_smoothing"
        )
        parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")
        parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 1.0)")
        parser.add_argument("--lr-scheduler", default="cosa", type=str, help="the lr scheduler (default: cosa)")
        parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 5)")
        parser.add_argument(
            "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)"
        )
        parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
        parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
        parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
        parser.add_argument("--output-dir", default="./logs", type=str, help="path to save outputs")
        parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
        parser.add_argument(
            "--cache-dataset",
            dest="cache_dataset",
            help="Cache the datasets for quicker initialization. It also serializes the transforms",
            action="store_true",
        )
        parser.add_argument(
            "--sync-bn",
            dest="sync_bn",
            help="Use sync batch norm",
            action="store_true",
        )
        parser.add_argument(
            "--test-only",
            dest="test_only",
            help="Only test the model",
            action="store_true",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            help="Use pre-trained models from the modelzoo",
            action="store_true",
        )
        parser.add_argument("--auto-augment", default='ta_wide', type=str, help="auto augment policy (default: ta_wide)")
        parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")

        # Mixed precision training parameters

        parser.add_argument(
            "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
        )
        parser.add_argument(
            "--model-ema-steps",
            type=int,
            default=32,
            help="the number of iterations that controls how often to update the EMA model (default: 32)",
        )
        parser.add_argument(
            "--model-ema-decay",
            type=float,
            default=0.99998,
            help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
        )
        parser.add_argument(
            "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
        )
        parser.add_argument(
            "--val-resize-size", default=232, type=int, help="the resize size used for validation (default: 232)"
        )
        parser.add_argument(
            "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
        )
        parser.add_argument(
            "--train-crop-size", default=176, type=int, help="the random crop size used for training (default: 176)"
        )
        parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
        parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
        parser.add_argument(
            "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 4)"
        )

        # Prototype models only
        parser.add_argument(
            "--prototype",
            dest="prototype",
            help="Use prototype model builders instead those from main area",
            action="store_true",
        )
        parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
        parser.add_argument("--seed", default=2020, type=int, help="the random seed")

        parser.add_argument("--disable-pinmemory", action="store_true", help="not use pin memory in dataloader, which can help reduce memory consumption")
        parser.add_argument("--disable-amp", action="store_true",
                            help="not use automatic mixed precision training")
        parser.add_argument("--local_rank", type=int, help="args for DDP, which should not be set by user")
        parser.add_argument("--disable-uda", action="store_true",
                            help="not set 'torch.use_deterministic_algorithms(True)', which can avoid the error raised by some functions that do not have a deterministic implementation")

        # parser.add_argument('--T', type=int, help="total time-steps")
        parser.add_argument('--T', nargs='+', type=int, help="time steps per bitwidth model")
        parser.add_argument('--N', type=int, help="bitwidth of the spike")
        parser.add_argument('--R', type=int, help="number of repeats")
        parser.add_argument('--acc', type=float, help="target accuracy", default=0.8)
        parser.add_argument('--debug', action="store_true", help="debug mode")
        parser.add_argument('--dataset', type=str, default='CIFAR10', help="dataset name")
        parser.add_argument('--plot-from-data', action="store_true", help="plot from data, located in the output directory")
        parser.add_argument('--quantize', action="store_true", help="quantize the model")
        parser.add_argument('--bf16', action="store_true", help="use bfloat16")

        return parser
    
    def set_optimizer(self, args, parameters):
        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = None
        return optimizer
    
    def set_lr_scheduler(self, args, optimizer):
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "step":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosa":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs
            )
        elif args.lr_scheduler == "exp":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            main_lr_scheduler = None
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                warmup_lr_scheduler = None
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        return lr_scheduler

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{self.T}' + f'_N{args.N}'
    
    def load_data(self, args):
        func = getattr(self, f"load_{args.dataset}")
        return func(args)

    def load_model(self, args, num_classes=10):
        if args.model in models.__all__:
            model = models.__dict__[args.model](spiking_neuron=LIFNode, quantize=args.quantize, surrogate_function=Sigmoid(n=self.N), detach_reset=True)
            if args.bf16:
                model = model.bfloat16()
            # functional.set_step_mode(model, step_mode='m')
            if args.quantize:
                model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
                model = torch.quantization.prepare(model)
            
            self.train_nnz = [ [] for _ in range(model.num_spiking + 1) ]
            self.test_nnz = [ [] for _ in range(model.num_spiking + 1) ]

            return model
        else:
            raise ValueError(f"args.model should be one of {models.__all__}, got {args.model}")

    def main(self, args):
        # set_deterministic(args.seed, args.disable_uda)
        if args.prototype and prototype is None:
            raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
        if not args.prototype and args.weights:
            raise ValueError("The weights parameter works only in prototype mode. Please pass the --prototype argument.")

        # utils.init_distributed_mode(args)
        # print(args)

        device = torch.device(args.device)

        dataset, dataset_test, train_sampler, test_sampler = self.load_data(args)

        collate_fn = None
        num_classes = len(dataset.classes)
        if args.dataset in ['CIFAR10', 'FashionMNIST', 'MNIST']:
            mixup_transforms = []
            if args.mixup_alpha > 0.0:
                if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
                    pass
                else:
                    raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                            "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")
                mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
            if args.cutmix_alpha > 0.0:
                mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
            if mixup_transforms:
                mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
                collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=not args.disable_pinmemory
            )
        else:
            collate_fn = tonic.collation.PadTensors(batch_first=False)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=not args.disable_pinmemory, collate_fn=collate_fn
            )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=not args.disable_pinmemory,
            collate_fn=collate_fn
        )

        if args.debug:
            print("Creating model")
        model = self.load_model(args, num_classes)
        self.model = model
        model.to(device)
        # print(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if args.norm_weight_decay is None:
            parameters = model.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        optimizer = self.set_optimizer(args, parameters)

        if args.disable_amp:
            scaler = None
        else:
            scaler = torch.amp.GradScaler('cuda')

        lr_scheduler = self.set_lr_scheduler(args, optimizer)


        model_without_ddp = model

        model_ema = None
        if args.model_ema:
            # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

        if args.test_only:
            if model_ema:
                self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            else:
                self.evaluate(args, model, criterion, data_loader_test, device=device)
            return

        for epoch in range(args.start_epoch, args.epochs):

            start_time = time.time()

            self.before_train_one_epoch(args, model, epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)

            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
            if lr_scheduler is not None:
                lr_scheduler.step()

            self.before_test_one_epoch(args, model, epoch)
            test_loss, test_acc1, test_acc5 = self.evaluate(args, model, criterion, data_loader_test, device=device)

            if model_ema:
                ema_test_loss, ema_test_acc1, ema_test_acc5 = self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")

            print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

        if args.quantize:
            self.evaluate_quant(args, model, data_loader_test)
            print(f'Quantized test accuracy: {self.quant_acc1:.3f}')

    def load_CIFAR10(self, args):
        if args.debug:
            print("Loading data")

        data_path = os.path.join(args.data_path, args.dataset)

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            download=True
        )

        if args.debug:
            print("Took", time.time() - st)

        if args.debug:
            print("Loading validation data")

        test_dt = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            download=True
        )

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(train_dt)
        test_sampler = torch.utils.data.SequentialSampler(test_dt)

        return train_dt, test_dt, train_sampler, test_sampler
    
    def load_FashionMNIST(self, args):
        if args.debug:
            print("Loading data")

        data_path = os.path.join(args.data_path, args.dataset)

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = datasets.FashionMNIST(
            root=data_path,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ]),
            download=True
        )

        if args.debug:
            print("Took", time.time() - st)

        if args.debug:
            print("Loading validation data")

        test_dt = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ]),
            download=True
        )

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(train_dt)

        test_sampler = torch.utils.data.SequentialSampler(test_dt)

        return train_dt, test_dt, train_sampler, test_sampler
    
    def load_MNIST(self, args):
        if args.debug:
            print("Loading data")

        data_path = os.path.join(args.data_path, args.dataset)

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = datasets.MNIST(
            root=data_path,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ]),
            download=True
        )

        if args.debug:
            print("Took", time.time() - st)

        if args.debug:
            print("Loading validation data")

        test_dt = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ]),
            download=True
        )

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(train_dt)

        test_sampler = torch.utils.data.SequentialSampler(test_dt)

        return train_dt, test_dt, train_sampler, test_sampler
    
    def load_NMNIST(self, args):
        if args.debug:
            print("Loading data")
    
        data_path = os.path.join(args.data_path, args.dataset)

        sensor_size = tonic.datasets.NMNIST.sensor_size
        transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.T),
        ])

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = tonic.datasets.NMNIST(data_path, transform=transform, train=True)
        disk_cached_train = tonic.DiskCachedDataset(train_dt, os.path.join(data_path, 'train_cache'))

        if args.debug:
            print("Took", time.time() - st)
        
        if args.debug:
            print("Loading validation data")

        test_dt = tonic.datasets.NMNIST(data_path, transform=transform, train=False)
        disk_cached_test = tonic.DiskCachedDataset(test_dt, os.path.join(data_path, 'test_cache'))

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(disk_cached_train)
        test_sampler = torch.utils.data.SequentialSampler(disk_cached_test)

        disk_cached_train.classes = [str(i) for i in range(10)]
        disk_cached_test.classes = [str(i) for i in range(10)]

        return disk_cached_train, disk_cached_test, train_sampler, test_sampler

    def load_DVSGesture(self, args):
        if args.debug:
            print("Loading data")

        data_path = os.path.join(args.data_path, args.dataset)

        sensor_size = tonic.datasets.DVSGesture.sensor_size
        transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.T),
        ])

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = tonic.datasets.DVSGesture(data_path, transform=transform, train=True)
        disk_cached_train = tonic.DiskCachedDataset(train_dt, os.path.join(data_path, 'train_cache'))

        if args.debug:
            print("Took", time.time() - st)
        
        if args.debug:
            print("Loading validation data")

        test_dt = tonic.datasets.DVSGesture(data_path, transform=transform, train=False)
        disk_cached_test = tonic.DiskCachedDataset(test_dt, os.path.join(data_path, 'test_cache'))

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(disk_cached_train)
        test_sampler = torch.utils.data.SequentialSampler(disk_cached_test)

        disk_cached_train.classes = [str(i) for i in range(11)]
        disk_cached_test.classes = [str(i) for i in range(11)]

        return disk_cached_train, disk_cached_test, train_sampler, test_sampler

    def load_CIFAR10DVS(self, args):
        if args.debug:
            print("Loading data")

        data_path = os.path.join(args.data_path, args.dataset)

        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
        transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.T),
        ])

        if args.debug:
            print("Loading training data")
        st = time.time()
        train_dt = tonic.datasets.CIFAR10DVS(data_path, transform=transform, train=True)
        disk_cached_train = tonic.DiskCachedDataset(train_dt, os.path.join(data_path, 'train_cache'))

        if args.debug:
            print("Took", time.time() - st)
        
        if args.debug:
            print("Loading validation data")

        test_dt = tonic.datasets.CIFAR10DVS(data_path, transform=transform, train=False)
        disk_cached_test = tonic.DiskCachedDataset(test_dt, os.path.join(data_path, 'test_cache'))

        if args.debug:
            print("Creating data loaders")

        train_sampler = torch.utils.data.RandomSampler(disk_cached_train)
        test_sampler = torch.utils.data.SequentialSampler(disk_cached_test)

        disk_cached_train.classes = [str(i) for i in range(10)]
        disk_cached_test.classes = [str(i) for i in range(10)]

        return disk_cached_train, disk_cached_test, train_sampler, test_sampler
        