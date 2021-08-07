import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms, utils
import os
from .invariance_generator import InvarianceGenerator
# from types import Optional, Callable, Tuple, Any

class AdversarialMNIST(MNIST):
    def __init__(self,
        root: str,
        adv_root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super(AdversarialMNIST, self).__init__(root, train, transform, target_transform, download)
        self.adv_root = adv_root

    def __getitem__(self, index: int):
        img, target = super(AdversarialMNIST, self).__getitem__(index)
        split = "train" if self.train else "test"
        adv_path = os.path.join(self.adv_root, f"inv_attacks_{split}_{index}.pt")

        # import pdb; pdb.set_trace();
        if os.path.exists(adv_path):
            adv_img = torch.load(os.path.join(self.adv_root, f"inv_attacks_{split}_{index}.pt"))
            adv_img = adv_img.unsqueeze(0)
        else:
            raise FileNotFoundError("Adversarial image not found.")
        
        return img, target, adv_img

    def __len__(self) -> int:
        return 60000 if self.train else 10000



class AdversarialCIFAR10(CIFAR10):
    def __init__(self,
        root: str,
        adv_root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super(AdversarialCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.adv_root = adv_root

    def __getitem__(self, index: int):
        img, target = super(AdversarialCIFAR10, self).__getitem__(index)
        split = "train" if self.train else "test"
        adv_path = os.path.join(self.adv_root, f"inv_attacks_cifar_{split}_{index}.pt")

        # import pdb; pdb.set_trace();
        if os.path.exists(adv_path):
            adv_img = torch.load(os.path.join(self.adv_root, f"inv_attacks_cifar_{split}_{index}.pt"))
        else:
            raise FileNotFoundError("Adversarial image not found.")
        
        # import pdb; pdb.set_trace();
        adv_img = adv_img.permute(2, 0, 1)
        return img, target, adv_img

    def __len__(self) -> int:
        return 50000 if self.train else 10000