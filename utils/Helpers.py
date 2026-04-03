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