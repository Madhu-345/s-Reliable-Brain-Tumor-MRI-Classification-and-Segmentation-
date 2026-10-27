"""
BraTS 2020 dataset loader.

Expected directory structure:

data/
└── BraTS2020/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii
    │   ├── BraTS20_Training_001_t1ce.nii
    │   ├── BraTS20_Training_001_t2.nii
    │   ├── BraTS20_Training_001_flair.nii
    │   └── BraTS20_Training_001_seg.nii
    ├── BraTS20_Training_002/
    └── ...

The loader returns:
    image: Tensor [C, H, W]
    mask:  Tensor [3, H, W]

Segmentation channels:
    channel 0 -> WT (Whole Tumor)
    channel 1 -> TC (Tumor Core)
    channel 2 -> ET (Enhancing Tumor)
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None


class BraTSDataset(Dataset):
    """
    BraTS 2020 slice-based dataset.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing BraTS patient folders.

    transform : callable, optional
        Optional transform applied to the image and mask.

    slice_axis : int
        Axis used to extract 2-D slices from the 3-D volumes.

    modalities : Sequence[str]
        MRI modalities to load.

    include_empty_slices : bool
        If False, slices without tumor are excluded.

    image_size : int, optional
        If provided, images and masks are resized to this size.
    """

    MODALITY_SUFFIXES = {
        "T1": "_t1.nii",
        "T1ce": "_t1ce.nii",
        "T2": "_t2.nii",
        "FLAIR": "_flair.nii",
    }

    SEGMENTATION_SUFFIX = "_seg.nii"

    def __init__(
        self,
        root_dir,
        transform: Optional[object] = None,
        slice_axis: int = 2,
        modalities: Sequence[str] = ("T1", "T1ce", "T2", "FLAIR"),
        include_empty_slices: bool = True,
        image_size: Optional[int] = None,
    ):
        if nib is None:
            raise ImportError(
                "nibabel is required for BraTS data loading. "
                "Install it using: pip install nibabel"
            )

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.slice_axis = slice_axis
        self.modalities = list(modalities)
        self.include_empty_slices = include_empty_slices
        self.image_size = image_size

        self._validate_inputs()

        self.patient_dirs = sorted(
            [
                path
                for path in self.root_dir.iterdir()
                if path.is_dir()
            ]
        )

        if not self.patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in: {self.root_dir}"
            )

        self.samples = self._build_slice_index()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_inputs(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"BraTS directory does not exist: {self.root_dir}"
            )

        if self.slice_axis not in (0, 1, 2):
            raise ValueError(
                "slice_axis must be 0, 1, or 2."
            )

        valid_modalities = set(self.MODALITY_SUFFIXES.keys())

        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(
                    f"Unsupported modality '{modality}'. "
                    f"Supported modalities: {sorted(valid_modalities)}"
                )

        if len(self.modalities) == 0:
            raise ValueError(
                "At least one MRI modality must be selected."
            )

    # --------------------------------------------------------
    # File utilities
    # --------------------------------------------------------

    @staticmethod
    def _find_file(patient_dir: Path, suffix: str) -> Optional[Path]:
        """
        Find a NIfTI file ending with the requested suffix.

        Supports both .nii and .nii.gz.
        """

        candidates = list(patient_dir.glob(f"*{suffix}"))

        if not candidates and suffix.endswith(".nii"):
            candidates = list(
                patient_dir.glob(f"*{suffix}.gz")
            )

        if not candidates:
            return None

        return candidates[0]

    # --------------------------------------------------------
    # Slice index
    # --------------------------------------------------------

    def _build_slice_index(self):
        samples = []

        for patient_dir in self.patient_dirs:

            modality_files = {}

            valid_patient = True

            for modality in self.modalities:

                file_path = self._find_file(
                    patient_dir,
                    self.MODALITY_SUFFIXES[modality],
                )

                if file_path is None:
                    valid_patient = False
                    break

                modality_files[modality] = file_path

            seg_file = self._find_file(
                patient_dir,
                self.SEGMENTATION_SUFFIX,
            )

            if not valid_patient or seg_file is None:
                continue

            try:
                seg_volume = nib.load(str(seg_file)).get_fdata()
            except Exception:
                continue

            num_slices = seg_volume.shape[self.slice_axis]

            for slice_idx in range(num_slices):

                if self.include_empty_slices:
                    samples.append(
                        {
                            "patient": patient_dir.name,
                            "slice": slice_idx,
                            "modalities": modality_files,
                            "segmentation": seg_file,
                        }
                    )

                else:
                    seg_slice = self._extract_slice(
                        seg_volume,
                        slice_idx,
                    )

                    if np.any(seg_slice > 0):
                        samples.append(
                            {
                                "patient": patient_dir.name,
                                "slice": slice_idx,
                                "modalities": modality_files,
                                "segmentation": seg_file,
                            }
                        )

        return samples

    # --------------------------------------------------------
    # Slice extraction
    # --------------------------------------------------------

    def _extract_slice(self, volume, slice_idx):

        if self.slice_axis == 0:
            return volume[slice_idx, :, :]

        if self.slice_axis == 1:
            return volume[:, slice_idx, :]

        return volume[:, :, slice_idx]

    # --------------------------------------------------------
    # Intensity normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize(volume):
        """
        Z-score normalization using non-zero voxels.
        """

        volume = volume.astype(np.float32)

        mask = volume != 0

        if np.any(mask):

            values = volume[mask]

            mean = values.mean()
            std = values.std()

            if std > 1e-8:
                volume[mask] = (
                    volume[mask] - mean
                ) / std
            else:
                volume[mask] = (
                    volume[mask] - mean
                )

        return volume

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    def _resize(self, image, mask):
        """
        Resize 2-D slices.

        Image uses bilinear interpolation.
        Mask uses nearest-neighbor interpolation.
        """

        if self.image_size is None:
            return image, mask

        import torch.nn.functional as F

        image_tensor = torch.from_numpy(
            image
        ).float().unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask
        ).float().unsqueeze(0)

        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        image = image_tensor.squeeze(0).numpy()
        mask = mask_tensor.squeeze(0).numpy()

        return image, mask

    # --------------------------------------------------------
    # Mask construction
    # --------------------------------------------------------

    @staticmethod
    def _create_regions(segmentation):
        """
        Convert BraTS labels into three binary regions.

        BraTS labels:
            1 -> Necrotic / non-enhancing tumor
            2 -> Edema
            4 -> Enhancing tumor

        Regions:
            WT = labels 1, 2, 4
            TC = labels 1, 4
            ET = label 4
        """

        wt = np.isin(
            segmentation,
            [1, 2, 4],
        )

        tc = np.isin(
            segmentation,
            [1, 4],
        )

        et = segmentation == 4

        mask = np.stack(
            [
                wt,
                tc,
                et,
            ],
            axis=0,
        )

        return mask.astype(np.float32)

    # --------------------------------------------------------
    # Load sample
    # --------------------------------------------------------

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load MRI modalities
        # ----------------------------------------------------

        modality_slices = []

        reference_shape = None

        for modality in self.modalities:

            volume = nib.load(
                str(sample["modalities"][modality])
            ).get_fdata()

            if reference_shape is None:
                reference_shape = volume.shape

            if volume.shape != reference_shape:
                raise ValueError(
                    f"Volume shape mismatch for patient "
                    f"{sample['patient']}."
                )

            slice_data = self._extract_slice(
                volume,
                sample["slice"],
            )

            slice_data = self._normalize(
                slice_data
            )

            modality_slices.append(
                slice_data
            )

        image = np.stack(
            modality_slices,
            axis=0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # Load segmentation
        # ----------------------------------------------------

        segmentation = nib.load(
            str(sample["segmentation"])
        ).get_fdata()

        seg_slice = self._extract_slice(
            segmentation,
            sample["slice"],
        )

        mask = self._create_regions(
            seg_slice
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        if self.image_size is not None:

            resized_images = []

            for channel in image:

                channel_tensor = torch.from_numpy(
                    channel
                ).float().unsqueeze(0).unsqueeze(0)

                channel_tensor = torch.nn.functional.interpolate(
                    channel_tensor,
                    size=(
                        self.image_size,
                        self.image_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                resized_images.append(
                    channel_tensor.squeeze().numpy()
                )

            image = np.stack(
                resized_images,
                axis=0,
            )

            mask_tensor = torch.from_numpy(
                mask
            ).float().unsqueeze(0)

            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="nearest",
            )

            mask = mask_tensor.squeeze(0).numpy()

        # ----------------------------------------------------
        # Optional transform
        # ----------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            np.asarray(image)
        ).float()

        mask = torch.from_numpy(
            np.asarray(mask)
        ).float()

        return {
            "image": image,
            "mask": mask,
            "patient": sample["patient"],
            "slice": sample["slice"],
        }

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)