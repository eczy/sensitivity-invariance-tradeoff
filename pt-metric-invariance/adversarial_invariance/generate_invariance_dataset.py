import argparse
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from invariance_generator import InvarianceGenerator
import os
import shutil
import numpy as np
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist_dir", default="./mnist")
    args = parser.parse_args()

    mnist_train = MNIST(args.mnist_dir, train=True, download=True)
    mnist_test = MNIST(args.mnist_dir, train=False, download=True)

    for split, ds in zip(['train', 'test'], [mnist_test, mnist_train]):
        print(f"Generating attacks for {split}")
        inv_generator = InvarianceGenerator()
        inv_generator.fit(ds.data.numpy(), np.array([y for x, y in ds]))

        for i, (x, y) in enumerate(tqdm(ds)):
            if os.path.exists(os.path.join(args.mnist_dir, f"inv_attacks_{split}_{i}.pt")):
                continue
            attack = torch.Tensor(inv_generator.invariance_attack(np.array(x), y))
            torch.save(attack, os.path.join(args.mnist_dir, f"inv_attacks_{split}_{i}.pt"))
            del attack

        # attacks_tensor = torch.stack(attacks)
        # torch.save(attacks_tensor, os.path.join(args.mnist_dir, f"inv_attacks_{split}.pt"))


if __name__ == "__main__":
    main()