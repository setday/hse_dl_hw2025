from typing import Optional

import torchvision.transforms.functional as F
from datasets import load_dataset

from src.datasets.base_dataset import BaseDataset


class SidSetDataset(BaseDataset):
    def __init__(self,
                 part: str, 
                 limit: Optional[int] = None,
                 instance_transforms=None):
        assert part in ["train:classification", "val:classification",
                        "train:segmentation", "val:segmentation"], \
            "part must be one of 'train:classification', 'val:classification', " \
            "'train:segmentation', 'val:segmentation'"
        
        self.mode, self.task = part.split(':')

        self.instance_transforms = instance_transforms

        self.limit = limit or {
            "train": 210000,
            "val": 30000,
        }[self.mode]
        self.num_classes = {
            "classification": 3,
            "segmentation": 1,
        }[self.task]

        self.index = load_dataset(
            "saberzl/SID_Set",
            split="train" if "train" == self.mode else "validation",
            num_proc=16,
        )

        if self.task == "segmentation":
            self.index = self.index.filter(lambda x: x["label"] == 2, num_proc=16)
            self.limit = min(self.limit, len(self.index))


    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self.index[ind]
        image_path = data_dict.get("__url__", None)
        
        if "image" in data_dict:
            image = data_dict["image"]
        else:
            raise ValueError("No image found in the dataset item.")
        
        if self.task == "classification" and "label" in data_dict:
            target = data_dict["label"]
        elif self.task == "segmentation" and "mask" in data_dict:
            target = (F.pil_to_tensor(data_dict["mask"].convert("L")) / 255.0).long()
        else:
            raise ValueError("No label found in the dataset item.")

        instance_data = {
            "image": F.pil_to_tensor(image.convert("RGB")),
            "target": target,

            "image_path": image_path,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return self.limit

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data
