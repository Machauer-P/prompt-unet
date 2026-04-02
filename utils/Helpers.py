import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import tensorflow_probability as tfp
# import torch
import mlflow
from scipy.ndimage import zoom
from scipy.ndimage import map_coordinates

class Helpers():
    """
    class with help functions for testing and visualization.
    """
    
    def __init__(self):
        pass
    
    # --- Other ---
    
    def shaping(self, tensor, h=128, w=128, binary=False):
        """Ensure proper shape (1, 128, 128, 1/2) of tf tensor.
        """
        if len(tensor.shape) == 3:
            # (128,128,1/2)
            if tensor.shape[0] > 1 and tensor.shape[1] > 1 and (tensor.shape[2] == 1 or tensor.shape[2] == 2):
                tensor = tensor[tf.newaxis,...]
            # (1,128,128)
            elif tensor.shape[0] == 1 and tensor.shape[1] > 1 and tensor.shape[2] > 1:
                tensor = tensor[...,tf.newaxis]

        if len(tensor.shape) == 2:
            tensor = tensor[tf.newaxis,...,tf.newaxis]

        if tensor.shape[1] != h or tensor.shape[2] != w:
            if binary:
                tensor = tf.image.resize(tensor, [h, w], method='nearest')
            else:
                tensor = tf.image.resize(tensor, [h, w])

        if tensor.shape == (1,h,w,1) or tensor.shape == (1,h,w,2):
            pass
        else:
            raise Exception(f'Something went wrong. Shape is {tensor.shape}.')

        return tensor
    
#     def shaping(self, tensor, binary=False):
#         """Ensure proper shape (1, 128, 128, 1/2) of tf tensor.
#         """
#         if len(tensor.shape) == 3:
#             # (128,128,1/2)
#             if tensor.shape[0] > 1 and tensor.shape[1] > 1 and (tensor.shape[2] == 1 or tensor.shape[2] == 2):
#                 tensor = tensor[tf.newaxis,...]
#             # (1,128,128)
#             elif tensor.shape[0] == 1 and tensor.shape[1] > 1 and tensor.shape[2] > 1:
#                 tensor = tensor[...,tf.newaxis]

#         if len(tensor) == 2:
#             tensor = tensor[tf.newaxis,...,tf.newaxis]

#         if tensor.shape[1] != 128 or tensor.shape[2] != 128:
#             tensor = tf.image.resize(tensor, [128, 128])
#             if binary:
#                 tensor = tf.cast(tensor != 0, tf.float32)

#         if tensor.shape == (1,128,128,1) or tensor.shape == (1,128,128,2):
#             pass
#         else:
#             raise Exception(f'Something went wrong. Shape is {tensor.shape}.')

#         return tensor
    
    
    
    
    def min_max_norm(self, image, lower_q=0.5, upper_q=99.5):
        image = tf.cast(image, tf.float32)
        flat = tf.reshape(image, [-1])

        # Berechne Quantile
        q_min = tfp.stats.percentile(flat, lower_q, interpolation='nearest')
        q_max = tfp.stats.percentile(flat, upper_q, interpolation='nearest')

        # Clip nur innerhalb der Quantile (robust)
        image = tf.clip_by_value(image, q_min, q_max)

        # Min–Max Normalisierung
        image = (image - q_min) / (q_max - q_min + 1e-8)
        return image

    
    # def min_max_norm_per_slice(self, tensor, epsilon=1e-7):
    #     "Used if tensor looks like (num_slices, H, W)."
    #     tensor = tf.cast(tensor, tf.float32)
    #     min_val = tf.reduce_min(tensor, axis=[1,2], keepdims=True)
    #     max_val = tf.reduce_max(tensor, axis=[1,2], keepdims=True)
    #     return (tensor - min_val) / (max_val - min_val + epsilon)
    
    # --- Visualization ---
    
    def plot_one_dp(self, x, y, p, offset, contrast=1):
        fig, axes = plt.subplots(1, 2, figsize=(7, 7))

        y = np.squeeze(y.numpy())
        x = np.squeeze(x.numpy())
        p = np.squeeze(p.numpy())

        # Binär/Contrast setzen
        y_mask = (y > 0)
        y_img = np.stack([x, x, x], axis=-1)  # RGB
        y_img[y_mask] = [1, 1, 0]  # Gelb

        axes[0].imshow(y_img)
        axes[0].set_title('Query (x) + Target (y)')
        axes[0].axis("off") 

        # Prompt visualisieren
        p1 = p[..., 1]
        p1_mask = (p1 > 0)

        p_img = np.stack([p[..., 0], p[..., 0], p[..., 0]], axis=-1)
        p_img[p1_mask] = [1, 1, 0]  # Gelb

        axes[1].imshow(p_img)
        axes[1].set_title(f'Prompt (offset = {offset})')
        axes[1].axis("off") 

        plt.show()
        print()
        
    def plot_result(self, x, y, p, pred, offset, pred_titel='', contrast=1):
        fig, axes = plt.subplots(1, 3, figsize=(7, 7))

        y = np.squeeze(y.numpy())
        x = np.squeeze(x.numpy())

        # --- Query + Target (gelb) ---
        y_mask = (y > 0)
        y_img = np.stack([x, x, x], axis=-1)
        y_img[y_mask] = [1, 1, 0]  # gelb

        axes[0].imshow(y_img)
        axes[0].set_title('Query (x) + Target (y)', fontsize=10)
        axes[0].axis("off")

        # --- Prompt (gelb) ---
        p = np.squeeze(p.numpy())
        p1 = p[..., 1]
        p1_mask = (p1 > 0)

        p_img = np.stack([p[..., 0], p[..., 0], p[..., 0]], axis=-1)
        p_img[p1_mask] = [1, 1, 0]  # gelb

        axes[1].imshow(p_img)
        axes[1].set_title(f'Prompt (offset = {offset})', fontsize=10)
        axes[1].axis("off")

        # --- Prediction ---
        axes[2].imshow(np.squeeze(pred))
        axes[2].set_title(pred_titel, fontsize=10)
        axes[2].axis("off")

        plt.show()
            
    def visualize_a_few_results(self, model_name:str, loaded_model: tf.keras.Model, ds, offset, img_to_plot=8, threshold=0.45, contrast=1):
        for i, (x,y,p) in enumerate(ds):
            if i == img_to_plot:
                break
                
            x = tf.expand_dims(x, axis=0) 
            p = tf.expand_dims(p, axis=0)
            pred = loaded_model.predict([x[0:1,:,:,0:1], p[0:1,...]])
            pred = tf.where(pred < threshold, 0.0, pred)
            pred = tf.where(pred >= threshold, 1.0, pred)

            self.plot_result(x,y,p,pred,offset[i],f'Prediction (Number {str(i)})', contrast)
            y = tf.cast(y, tf.float32)
            print(f"Dice: {self.dice_score_tf(y[..., 0:1], pred):.3f}\n")
    
    def plot_random_slice_from_vol(dataset, num_examples=9):
        """
        Takes a tf.data.Dataset of shape (num_slices, height, width)
        and plots one random slice from each Dataset.

        Args:
            dataset: tf.data.Dataset
            num_examples: number of dataset elements to sample and plot
        """
        plt.figure(figsize=(10, 10))

        for i, (x, y) in enumerate(dataset.take(num_examples)):
            # volume shape: (num_slices, height, width)
            num_slices = tf.shape(x)[0]

            # Choose a random slice index
            rand_idx = tf.random.uniform([], minval=0, maxval=num_slices, dtype=tf.int32)
            slice_img = x[rand_idx, :, :]
            slice_segm = y[rand_idx, :, :]

            # Convert to numpy for plotting
            slice_img = slice_img.numpy()
            slice_segm = slice_segm.numpy()

            plt.subplot(int(np.ceil(np.sqrt(num_examples))), int(np.ceil(np.sqrt(num_examples))), i + 1)
            plt.imshow(slice_img + slice_segm)
            plt.axis('off')
            plt.title(f"Slice {rand_idx.numpy()}")

        plt.tight_layout()
        plt.show()
    
#     def plot_samples_from_vol(self, dataset, idx_list, num_img=10, max_entries=300):
#         """
#         Plots `num_img` images evenly spaced from a TensorFlow dataset of (image, label) pairs or a tuple of (vol_img, vol_labels).

#         Args:
#             dataset (tf.data.Dataset): Dataset containing (image, label) pairs or a tuple of (vol_img, vol_labels).
#             idx_list (list): List with the index of the current slice. Needs to match with dataset. 
#             num_img (int): Number of images to plot.
#             max_entries (int): Limit to this many samples for faster plotting.
#         """
#         if type(dataset) == tf.data.Dataset:

#             count = sum(1 for _ in dataset.take(max_entries))
#             if count == 0:
#                 print("Dataset is empty.")
#                 return

#             if count < num_img:
#                 num_img = count

#             indices = [int(i * count / num_img) for i in range(num_img)]

#             plt.figure(figsize=(num_img * 3, 3))
#             for idx, (image, label) in enumerate(dataset.take(max_entries)):
#                 if idx in indices:
#                     plot_idx = indices.index(idx) + 1
#                     plt.subplot(1, num_img, plot_idx)
#                     plt.imshow(image.numpy().squeeze() + label.numpy().squeeze())
#                     plt.title(str(idx_list[idx]), fontsize=14)
#                     plt.axis("off")

#             plt.tight_layout()
#             plt.show()

#         elif type(dataset) == tuple:

#             vol, labels = dataset

#             count = vol.shape[0] 

#             if count < num_img:
#                 num_img = count

#             indices = [int(i * vol.shape[0] / num_img) for i in range(num_img)]

#             plt.figure(figsize=(num_img * 3, 3))
#             # First dim == img counter
#             for i in indices:
#                 image = vol[i,...]
#                 label = labels[i,...]

#                 plot_idx = indices.index(i) + 1
#                 plt.subplot(1, num_img, plot_idx)
#                 plt.imshow(image.numpy().squeeze() + label.numpy().squeeze())
#                 plt.title(str(idx_list[i]), fontsize=14)
#                 plt.axis("off")

#             plt.tight_layout()
#             plt.show()

#         else:
#             return


    # --- Metrics ---
    
    # Make one np versions that workd with tf and torch!
    def dice_score_tf(self, y_true, y_pred, smooth=1e-6):
        if (y_true.dtype) != tf.float32:
            y_true = tf.cast(y_true, tf.float32)
        if (y_pred.dtype) != tf.float32:
            y_pred = tf.cast(y_pred, tf.float32)

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        return dice
    
     # Assume numpy arrays
    def dice_numpy(self, y_true, y_pred, smooth=1e-6):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        # Flatten
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)

        # Dice calculation
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
        return dice  # stays as numpy float
    
    
    # def dice_score_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    #     y_pred = y_pred.long()
    #     y_true = y_true.long()
    #     score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum() + 1)
    #     return score.item()
    
    # --- Testing ---

    def test_routine(self, model_name: str, loaded_model: tf.keras.Model, ds, offset, threshold=0.45, ml_flow=False):
        
        start = time.time()
        total_dice = 0
        len_ds = len(list(ds.as_numpy_iterator()))

        if ml_flow:
            mlflow.set_experiment(f'{model_name}_testing')
            with mlflow.start_run() as run:
                mlflow.log_param("binarization_threshold", threshold)
                mlflow.log_param("max_offset", f'+/-{offset}')

                for e, (x, y, p) in enumerate(ds):
                    
                    x = self.shaping(x)
                    p = self.shaping(p)
                    
                    pred = loaded_model.predict([x[0:1, :, :, 0:1], p[0:1, ...]], verbose=0)

                    pred = tf.where(pred < threshold, 0.0, pred)
                    pred = tf.where(pred >= threshold, 1.0, pred)
                    current_dice = self.dice_score_tf(y[..., 0:1], pred)

                    mlflow.log_metric("dice_score", current_dice, step=e)
                    total_dice += current_dice

                mlflow.log_metric("avg_dice_score", total_dice)

        else:
            for e, (x, y, p) in enumerate(ds):
                
                x = self.shaping(x)
                p = self.shaping(p)
                    
                pred = loaded_model.predict([x[0:1, :, :, 0:1], p[0:1, ...]], verbose=0)

                pred = tf.where(pred < threshold, 0.0, pred)
                pred = tf.where(pred >= threshold, 1.0, pred)
                current_dice = self.dice_score_tf(y[..., 0:1], pred)
                total_dice += current_dice

        end = time.time()
        print(f'Testing for {model_name} took {round(end - start, 0)} seconds')

        return (total_dice / len_ds).numpy()
    
    
    
    # --- Resampling ---
    
    def resample_isotropic(self, volume, voxel_sizes, new_voxel_size=None, order=1):
        """
        Resample 3D-Volume to isotropic resolution.

        volume: numpy array
        voxel_sizes: Tuple (sx, sy, sz) in mm
        new_voxel_size: gewünschte isotrope Größe in mm
        order: choose 0 for masks and 1 for images
        """
        if new_voxel_size is None:
            new_voxel_size = min(voxel_sizes)  # kleinster Wert für max. Auflösung

        # Zoom-Faktoren berechnen
        zoom_factors = [vs / new_voxel_size for vs in voxel_sizes]

        # Resampling (linear)
        volume_iso = zoom(volume, zoom=zoom_factors, order=order)  
        volume_iso = tf.convert_to_tensor(volume_iso)
        return volume_iso
    
    def gen_grid(self, shape):
        # slice_size = max(shape)
        slice_size = int(np.ceil(np.linalg.norm(shape)))
        x_grid = np.linspace(-slice_size//2, slice_size//2, slice_size)
        y_grid = np.linspace(-slice_size//2, slice_size//2, slice_size)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        return grid_x, grid_y

    def __random_plane_coords(self, shape, center:np.array, center_offset:np.array, grid_x, grid_y):
        """Erzeugt zufällige Schnittebene und liefert die 3D-Koordinaten für jeden Pixel.
        """
        # int(np.ceil(np.linalg.norm(shape))) macht Bild größer, deswegen max(shape). Es gibt das Risiko, dass etwas abgeschnitten wird, je nach Perspektive
        # slice_size = max(shape)
        slice_size = int(np.ceil(np.linalg.norm(shape)))  # Raumdiagonale

        # Zufälliger Mittelpunkt in Voxelkoordinaten
        # center = np.array([np.random.uniform(0, shape[0]),
        #                    np.random.uniform(0, shape[1]),
        #                    np.random.uniform(0, shape[2])])

        # Zufällige Normale (Richtung der Ebene)
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal) # Einheitsvektor erzeugen

        # Zwei Vektoren in der Ebene finden, um Ebene aufzuspannen
        v1 = np.random.randn(3) # zufälliger Vektor
        v1 -= v1.dot(normal) * normal # Anteil entlang der Normalen entfernen
        v1 /= np.linalg.norm(v1) # Einheitsvektor
        v2 = np.cross(normal, v1) # Senkrechter Vektor erzeugen
        # v1 = x-Richtung und v2 = y-Richtung der Ebene
        # Beide liegen in der Ebene, sind rechtwinklig und haben Länge 1

        # Vektorisierte Koordinatenberechnung (kein for-loop!)
        coords = np.zeros((3, slice_size, slice_size))  # Array, in dem die 3D-Koord. für jedes 2D-Pixel gepeichert wird. Dimension 0 = welche Koordinate (x, y, z). Dim 1,2 = Pixelpos.
        coords[0] = center[0] + grid_x * v1[0] + grid_y * v2[0]
        coords[1] = center[1] + grid_x * v1[1] + grid_y * v2[1]
        coords[2] = center[2] + grid_x * v1[2] + grid_y * v2[2]

        coords_offset = np.zeros((3, slice_size, slice_size))  # Array, in dem die 3D-Koord. für jedes 2D-Pixel gepeichert wird. Dimension 0 = welche Koordinate (x, y, z). Dim 1,2 = Pixelpos.
        coords_offset[0] = center_offset[0] + grid_x * v1[0] + grid_y * v2[0]
        coords_offset[1] = center_offset[1] + grid_x * v1[1] + grid_y * v2[1]
        coords_offset[2] = center_offset[2] + grid_x * v1[2] + grid_y * v2[2]

        return coords, coords_offset

    def random_plane_slice(self, volume_img, volume_seg, center:np.array, center_offset:np.array, grid_x, grid_y):
        """Randomly slices the volume image and semgementation in the same way. Outputs different planes than just x,y,z.
        Outputs: 
        slice_img, slice_seg: One Datapoint
        slice_img_offset, slice_seg_offset: Datapoint of the same new random volume but with a offset 
        """
        coords, coords_offset = self.__random_plane_coords(volume_img.shape, center, center_offset, grid_x, grid_y)

        # Image interpolation 
        slice_img = map_coordinates(volume_img, coords, order=1, mode='constant')
        slice_img_offset = map_coordinates(volume_img, coords_offset, order=1, mode='constant')

        # Segmentation interpolation
        slice_seg = map_coordinates(volume_seg, coords, order=0, mode='constant')
        slice_seg_offset = map_coordinates(volume_seg, coords_offset, order=0, mode='constant')
        
        def unified_crop(img1, seg1, img2, seg2):
            # Create a combined mask from all four arrays
            combined_mask = (img1 > 0) | (seg1 > 0) | (img2 > 0) | (seg2 > 0)

            if not np.any(combined_mask):
                # Return center portion if no content
                center_y, center_x = img1.shape[0] // 2, img1.shape[1] // 2
                crop_size = min(img1.shape) // 3
                ymin = max(0, center_y - crop_size // 2)
                ymax = min(img1.shape[0], center_y + crop_size // 2)
                xmin = max(0, center_x - crop_size // 2)
                xmax = min(img1.shape[1], center_x + crop_size // 2)
                return ymin, ymax, xmin, xmax

            # Find bounding box of combined content
            rows = np.any(combined_mask, axis=1)
            cols = np.any(combined_mask, axis=0)

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            ymin = max(0, ymin)
            ymax = min(img1.shape[0], ymax)
            xmin = max(0, xmin)
            xmax = min(img1.shape[1], xmax)

            return ymin, ymax, xmin, xmax

        # Get unified cropping coordinates
        ymin, ymax, xmin, xmax = unified_crop(slice_img, slice_seg, slice_img_offset, slice_seg_offset)

        # Apply same crop to all slices
        slice_img = slice_img[ymin:ymax, xmin:xmax]
        slice_seg = slice_seg[ymin:ymax, xmin:xmax]
        slice_img_offset = slice_img_offset[ymin:ymax, xmin:xmax]
        slice_seg_offset = slice_seg_offset[ymin:ymax, xmin:xmax]
        
        slice_img = tf.convert_to_tensor(slice_img)
        slice_seg = tf.convert_to_tensor(slice_seg)
        slice_img_offset = tf.convert_to_tensor(slice_img_offset)
        slice_seg_offset = tf.convert_to_tensor(slice_seg_offset)

        return slice_img, slice_seg, slice_img_offset, slice_seg_offset