import numpy as np
import os

def averaging(args):
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    firerate_train = np.load(os.path.join(data_dir, 'firerate_train.npy'))
    firerate_test = np.load(os.path.join(data_dir, 'firerate_test.npy'))

    tmp = firerate_train
    firerate_train = np.empty((*(tmp.shape[:-1]), tmp.shape[-1]//args.T[0]), dtype=float)
    for bit in range(tmp.shape[0]):
        for spiking in range(tmp.shape[1]):
            for run in range(tmp.shape[2]):
                for t in range(tmp.shape[3]//args.T[bit]):
                    firerate_train[bit, spiking, run, t] = np.mean(tmp[bit, spiking, run, t*args.T[bit]:(t+1)*args.T[bit]])
    
    np.save(os.path.join(data_dir, 'firerate_train.npy'), firerate_train)

    tmp = firerate_test
    firerate_test = np.empty((*(tmp.shape[:-1]), tmp.shape[-1]//args.T[0]), dtype=float)
    for bit in range(tmp.shape[0]):
        for spiking in range(tmp.shape[1]):
            for run in range(tmp.shape[2]):
                for t in range(tmp.shape[3]//args.T[bit]):
                    firerate_test[bit, spiking, run, t] = np.mean(tmp[bit, spiking, run, t*args.T[bit]:(t+1)*args.T[bit]])

    np.save(os.path.join(data_dir, 'firerate_test.npy'), firerate_test)
