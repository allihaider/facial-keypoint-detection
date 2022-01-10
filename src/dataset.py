import numpy as np
import tensorflow as tf
from imgaug.augmentables.kps import Keypoint
from imgaug.augmentables.kps import KeypointsOnImage


class KeyPointsDataset(tf.keras.utils.Sequence):
    
    def __init__(self, images, keypoints, augmentations, batch_size, shuffle=True):
        
        self.images = images
        self.keypoints = keypoints
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        
        return
    
    def __len__(self):
        
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        images = [self.images[k] for k in indexes]
        keypoints = [self.keypoints[k] for k in indexes]
        images, keypoints = self.data_generation(images, keypoints)

        return (images, keypoints)

    def on_epoch_end(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        return
    
    def data_generation(self, images, keypoints):
        
        len_image_side = images[0].shape[0]
        len_keypoints = len(keypoints[0])
        batch_images = np.empty((self.batch_size, len_image_side, len_image_side, 3), dtype="float32")
        batch_keypoints = np.empty((self.batch_size, len_keypoints), dtype="float32")

        for i, (image, keypoint) in enumerate(zip(images, keypoints)):
            keypoint_pairs = []
            
            for j in np.arange(0, len(keypoint), 2):
                keypoint_pairs.append(Keypoint(x=int(keypoint[j]), y=int(keypoint[j+1])))
            
            # kps_obj = KeypointsOnImage(keypoint_pairs, shape=image.shape)
            (new_images, new_kps_obj) = self.augmentations(image=image, keypoints=kps_obj)
            batch_images[i] = new_images
            kp_temp = []

            for keypoint in new_kps_obj:
                kp_x = np.nan_to_num(keypoint.x)
                kp_y = np.nan_to_num(keypoint.y) 
                if kp_x < 0 or kp_x > 95:
                    kp_x = 0
                if kp_y < 0 or kp_y > 95:
                    kp_y = 0
                kp_temp.append(kp_x)
                kp_temp.append(kp_y)

            batch_keypoints[i] = kp_temp
        
        return batch_images, batch_keypoints
