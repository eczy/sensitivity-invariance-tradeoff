from torch.utils.data import Dataset, DataLoader, MNIST
from torchvision import transforms, utils
import os
from .invariance_generator import InvarianceGenerator

class AdversarialMNIST(MNIST):
    def __init__(self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        adv_root: str,
    ) -> None:
        super(AdversarialMNIST, self).__init__(root, train, transform, target_transform, download)
        self.adv_root = adv_root

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super(AdversarialMNIST, self).__getitem__(index)
        adv_path = os.path.join(self.adv_root, f"inv_attacks_{split}_{index}.pt")
        if os.path.exists(adv_path):
            adv_img = torch.load(os.path.join(self.adv_root, f"inv_attacks_{split}_{index}.pt"))
        else:
            raise FileNotFoundError("Adversarial image not found.")
        return img, target, adv_img

    def __len__(self) -> int:
        return 60000 if self.train else 1000

