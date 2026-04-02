from Data.DataLoader import DataLoader
import os
import pickle
import numpy as np
from collections import defaultdict

class DataLoader_pkl(DataLoader):
    """
    pkl_paths: Paths to .pkl files. Create the pkl with DSHandler.save_initial_ds()
    val_size: Percentage of Patients to be used for validation. Use 0.0 when no validation run for training is made.
    mode: No effect
    max_img: No effect
    """

    def __init__(self, pkl_paths, val_size, mode="", max_img=10000):
        # Resolve paths relative to the location of the script (project root)
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent  # folder containing DataLoader_pkl.py
        project_root = script_dir.parent

        self.pkl_paths = [str((project_root / p).resolve()) for p in pkl_paths]

        super().__init__(val_size=val_size, mode=mode, max_img=max_img)

    # --------------------------------------------------------------

    def _to_numpy(self, x):
        """Convert tensor, list, nib or array into numpy array."""
        import tensorflow as tf
        import nibabel as nib

        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        if hasattr(x, "get_fdata"):  # nibabel
            return x.get_fdata()
        if isinstance(x, list):
            return np.array(x)
        raise TypeError(f"Unsupported data type: {type(x)}")

    # --------------------------------------------------------------

    def _get_segmentation_list(self, segs):
        """
        Ensures a list of 3D numpy arrays is returned.
        Handles cases where data is stored as a single 4D array (Channels, H, W, D).
        """
        # 1. Standard List/Tuple Case
        if isinstance(segs, (list, tuple)):
            return [self._to_numpy(s) for s in segs]
        
        # 2. Dictionary Case
        if isinstance(segs, dict):
            return [self._to_numpy(s) for s in segs.values()]
        
        # 3. Single Array Case (The Problematic Part)
        arr = self._to_numpy(segs)
        
        # Check if it is a 4D array (Channel-first assumption)
        # If shape is (4, 128, 128, 128), we split it into 4 separate volumes.
        if arr.ndim == 4:
            # Assuming channel is the first dimension (C, H, W, D) or (C, Z, Y, X)
            # You might need to check if channel is last (axis=-1) depending on your data!
            # Here we assume axis 0 is the channel/segmentation index.
            return [arr[i] for i in range(arr.shape[0])]
            
        return [arr]

    # --------------------------------------------------------------

    def _pull_data(self):
        """
        Loads all .pkl files, namespaces PIDs, 
        and fills self.dataset in DataLoader format.
        """

        print("\nLoading PKL dataset(s)…")

        for pkl_path in self.pkl_paths:
            if not os.path.exists(pkl_path):
                print(f"WARNING: File does not exist: {pkl_path}")
                continue

            # Load pickle
            try:
                with open(pkl_path, "rb") as f:
                    pkl_data = pickle.load(f)
            except Exception as e:
                print(f"ERROR reading {pkl_path}: {e}")
                continue

            print(f"Loaded {len(pkl_data)} PIDs from {pkl_path}")

            prefix = os.path.splitext(os.path.basename(pkl_path))[0]  # file name
            count = 0

            # Convert each item
            for pid, item in pkl_data.items():
                if count >= self.max_img:
                    break
                count += 1

                pid = f"{prefix}_{pid}"   # namespace the pid

                if "image" not in item:
                    print(f"WARNING: PID {pid} has no 'image'")
                    continue

                img = self._to_numpy(item["image"])

                # Find segmentation(s)
                if "segmentations" in item:
                    seg_list = self._get_segmentation_list(item["segmentations"])
                elif "segmentation" in item:
                    seg_list = self._get_segmentation_list(item["segmentation"])
                else:
                    seg_list = []
                    print(f"WARNING: PID {pid} has no segmentation")

                # Store into dataset buffer
                self.dataset[pid] = {
                    "image": img,
                    "segmentations": seg_list
                }

        print(f"\nFinal dataset size: {len(self.dataset)} patients.\n")

    # --------------------------------------------------------------

    def _data_to_dict(self, tset, lset, iset):
        pass 
