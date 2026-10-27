"""
Brain tumor MRI classification dataset loader.

Expected directory structure:

data/
└── classification/
    ├── glioma/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── meningioma/
    │   ├── image1.jpg
    │   └── ...
    ├── pituitary/
    │   ├── image1.jpg
    │   └── ...
    └── no_tumor/
        ├── image1.jpg
        └── ...

Classes:
    0 -> glioma
    1 -> meningioma
    2 -> pituitary
    3 -> no_tumor
"""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

import torch
from torch.utils.data import Dataset


class BrainTumorClassificationDataset(Dataset):
    """
    Dataset class for brain tumor MRI classification.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing one folder per class.

    transform : callable, optional
        Image transformation pipeline.

    class_names : list, optional
        Ordered list of class names.

    extensions : tuple
        Supported image file extensions.

    """

    DEFAULT_CLASSES = [
        "glioma",
        "meningioma",
        "pituitary",
        "no_tumor",
    ]

    DEFAULT_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    def __init__(
        self,
        root_dir,
        transform: Optional[Callable] = None,
        class_names=None,
        extensions=None,
    ):

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.class_names = (
            class_names
            if class_names is not None
            else self.DEFAULT_CLASSES
        )

        self.extensions = (
            extensions
            if extensions is not None
            else self.DEFAULT_EXTENSIONS
        )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Classification dataset directory does not exist: "
                f"{self.root_dir}"
            )

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(
                self.class_names
            )
        }

        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No classification images were found in "
                f"{self.root_dir}"
            )

    # ========================================================
    # Collect image samples
    # ========================================================

    def _collect_samples(self):

        for class_name in self.class_names:

            class_dir = self.root_dir / class_name

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            class_index = self.class_to_idx[class_name]

            for image_path in sorted(
                class_dir.rglob("*")
            ):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in self.extensions:
                    continue

                self.samples.append(
                    (
                        image_path,
                        class_index,
                    )
                )

    # ========================================================
    # Load image
    # ========================================================

    @staticmethod
    def _load_image(image_path):

        image = Image.open(image_path)

        # Convert all images to RGB so that the input
        # consistently contains three channels.
        image = image.convert("RGB")

        return image

    # ========================================================
    # Get sample
    # ========================================================

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = self._load_image(
            image_path
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        else:

            # Basic fallback conversion if no transform
            # pipeline is supplied.
            image = torch.from_numpy(
                __import__("numpy").array(
                    image
                )
            ).permute(
                2, 0, 1
            ).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "path": str(image_path),
            "class_name": self.class_names[label],
        }

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Dataset information
    # ========================================================

    @property
    def num_classes(self):

        return len(self.class_names)

    # ========================================================
    # Class distribution
    # ========================================================

    def class_distribution(self):

        distribution = {
            class_name: 0
            for class_name in self.class_names
        }

        for _, label in self.samples:

            class_name = self.class_names[label]

            distribution[class_name] += 1

        return distribution