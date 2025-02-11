# Working in progress...

Documentation can be found in `thesis/thesis.pdf`.

## Installation

WIP

## Reproduction

### Accuracy, Iterations, and Energy

| Dataset | Launch command |
| --- | --- |
| Fashion MNIST | `python -m test --N 8 --R 10 --T 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
| MNIST | `python -m test --N 8 --R 10 --T 10 --acc 0.80 --model MNISTNet --data-path /scratch/zyi/codeSpace/data --dataset MNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
 NMNIST | `python -m test --N 8 --R 10 --T 10 --acc 0.80 --model NMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset NMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
| DVS Gesture | `python -m test --N 8 --R 10 --T 10 --acc 0.80 --model DVSGestureNet --data-path /scratch/zyi/codeSpace/data --dataset DVSGesture --batch-size 128 --opt adam --lr 1e-3 --lr-scheduler cosa --epochs 20 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes` |
| CIFAR-10 | `python -m test --N 4 --R 5 --T 10 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes` |

### Firing rate

| Dataset | Launch command |
| --- | --- |
| Fashion MNIST | `python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
| MNIST | `python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model MNISTNet --data-path /scratch/zyi/codeSpace/data --dataset MNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
 NMNIST | `python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model NMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset NMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
| DVS Gesture | `python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model DVSGestureNet --data-path /scratch/zyi/codeSpace/data --dataset DVSGesture --batch-size 128 --opt adam --lr 1e-3 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate` |
| CIFAR-10 | `python -m test --N 2 --R 5 --T 10 10 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 1000 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate` |

### Energy tradeoff

| Dataset | Launch command |
| --- | --- |
| Fashion MNIST | `python -m test --N 2 --R 10 --T 10 4 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/timesteps --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp` |
| CIFAR-10 | `python -m test --N 2 --R 5 --T 10 4 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/timesteps` |