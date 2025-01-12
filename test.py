from trainer import RandomTrainer

args = RandomTrainer().get_args_parser().parse_args()

num_runs = args.R

import plots
import tools

def train_and_plot(): 
    trainers = [ [ RandomTrainer(N=i+1, T=args.T[i], args=args) for j in range(num_runs) ] for i in range(args.N) ]

    for i in range(args.N):
        for j in range(num_runs):
            print(f"====================================\n   Training with {i+1} bit(s) - Run {j+1}   \n====================================")
            trainers[i][j].main(args)

    train_accs = [ [ trainer.train_acc1 for trainer in trainer_b ] for trainer_b in trainers ]
    plots.plot_accs_train(train_accs, args)

    test_accs = [ [ trainer.test_acc1 for trainer in trainer_b ] for trainer_b in trainers ]
    plots.plot_accs_test(test_accs, args)

    plots.plot_accs_final(test_accs, args, horizontal=True)

    train_iters = [ [ trainer.train_iters for trainer in trainer_b ] for trainer_b in trainers ]
    plots.plot_iters_train(train_iters, args, horizontal=True)

    test_iters = [ [ trainer.test_iters for trainer in trainer_b ] for trainer_b in trainers ]
    plots.plot_iters_test(test_iters, args, horizontal=True)

    num_spiking = trainers[0][0].model.num_spiking

    train_firerate = [ [ [ trainer.train_nnz[i] for trainer in trainer_b ] for i in range(num_spiking + 1) ] for trainer_b in trainers ]
    plots.plot_firerate_train(train_firerate, args)

    test_firerate = [ [ [ trainer.test_nnz[i] for trainer in trainer_b ] for i in range(num_spiking + 1) ] for trainer_b in trainers ]
    plots.plot_firerate_test(test_firerate, args)

    plots.plot_firerate_final(test_firerate, args, horizontal=True)

    plots.plot_energy_train_gpu(train_iters, args, horizontal=True)

    plots.plot_energy_test_nh(test_firerate, args, horizontal=True)

    if args.quantize:
        quant_accs = [ [ trainer.quant_acc1 for trainer in trainer_b ] for trainer_b in trainers ]
        plots.plot_accs_quant(quant_accs, args, horizontal=True)

def just_plot():
    plots.plot_accs_train(None, args)

    plots.plot_accs_test(None, args)

    plots.plot_accs_final(None, args, horizontal=True)

    plots.plot_iters_train(None, args, horizontal=True)

    plots.plot_iters_test(None, args, horizontal=True)

    # tools.averaging(args)

    plots.plot_firerate_train(None, args)

    plots.plot_firerate_test(None, args)

    plots.plot_firerate_final(None, args, horizontal=True)

    plots.plot_energy_train_gpu(None, args, horizontal=True)

    plots.plot_energy_test_nh(None, args, horizontal=True)

    if args.quantize:
        plots.plot_accs_quant(None, args, horizontal=True)

if args.plot_from_data:
    just_plot()
else:
    train_and_plot()

# trainer = RandomTrainer(N=2, args=args)
# trainer.main(args)

"""
Standard:
python -m test --N 8 --R 10 --T 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 8 --R 10 --T 10 --acc 0.80 --model MNISTNet --data-path /scratch/zyi/codeSpace/data --dataset MNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 8 --R 10 --T 10 --acc 0.80 --model NMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset NMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 8 --R 10 --T 10 --acc 0.80 --model DVSGestureNet --data-path /scratch/zyi/codeSpace/data --dataset DVSGesture --batch-size 128 --opt adam --lr 1e-3 --lr-scheduler cosa --epochs 20 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes
python -m test --N 4 --R 5 --T 10 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes

Plots:
python -m test --N 8 --R 10 --T 10 10 10 10 10 10 10 10 --dataset FashionMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/standard --plot-from-data
python -m test --N 8 --R 10 --T 10 10 10 10 10 10 10 10 --dataset MNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/standard --plot-from-data
python -m test --N 8 --R 10 --T 10 10 10 10 10 10 10 10 --dataset NMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/standard --plot-from-data
python -m test --N 8 --R 10 --T 10 10 10 10 10 10 10 10 --dataset DVSGesture --output-dir /scratch/zyi/codeSpace/MultibitSpikes/standard --plot-from-data
python -m test --N 4 --R 5 --T 10 10 10 10 10 10 10 10 --dataset CIFAR10 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/standard --plot-from-data

Firerate:
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model MNISTNet --data-path /scratch/zyi/codeSpace/data --dataset MNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model NMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset NMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model DVSGestureNet --data-path /scratch/zyi/codeSpace/data --dataset DVSGesture --batch-size 128 --opt adam --lr 1e-3 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate
python -m test --N 2 --R 5 --T 10 10 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 1000 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate

Plots:
python -m test --N 2 --R 10 --T 10 10 --dataset FashionMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --plot-from-data
python -m test --N 2 --R 10 --T 10 10 --dataset MNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --plot-from-data
python -m test --N 2 --R 10 --T 10 10 --dataset NMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --plot-from-data
python -m test --N 2 --R 10 --T 10 10 --dataset DVSGesture --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --plot-from-data
python -m test --N 2 --R 5 --T 10 10 --dataset CIFAR10 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/firerate --plot-from-data

Quantization:
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/bf16 --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp --bf16
python -m test --N 2 --R 10 --T 10 10 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/quantized --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp --quantize

Plots:
python -m test --N 2 --R 10 --T 10 10 --dataset FashionMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/bf16 --bf16 --plot-from-data
python -m test --N 2 --R 10 --T 10 10 --dataset FashionMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/quantized --quantize --plot-from-data

Reduced time steps:
python -m test --N 2 --R 10 --T 10 4 --acc 0.80 --model FashionMNISTNet --data-path /scratch/zyi/codeSpace/data --dataset FashionMNIST --batch-size 128 --opt adam --lr 2e-3 --lr-scheduler none --epochs 5 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/timesteps --mixup-alpha 0.0 --cutmix-alpha 0.0 --label-smoothing 0.0 --disable-amp
python -m test --N 2 --R 5 --T 10 4 --acc 0.80 --model CIFAR10Net --data-path /scratch/zyi/codeSpace/data --dataset CIFAR10 --batch-size 128 --opt adam --lr 1e-5 --lr-scheduler none --epochs 50 --lr-warmup-epochs 0 --output-dir /scratch/zyi/codeSpace/MultibitSpikes/timesteps

Plots:
python -m test --N 2 --R 10 --T 10 4 --dataset FashionMNIST --output-dir /scratch/zyi/codeSpace/MultibitSpikes/timesteps --plot-from-data
"""
