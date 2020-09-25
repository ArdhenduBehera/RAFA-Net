"""
Directory based data generator supporing image augmentation and pre-processing for multiple data streams.
Augmentation is identical across all streams (by passing a keras.preprocessing.image.ImageDataGenerator instance with the desired parameters),
but pre-processing is individual.

Created by Alexander Keidel @ 17.05.2018
Modified by Andrew Gidney @ 10.08.2018
"""

import numpy as np
from os import listdir
from os.path import isdir, join, isfile
import keras
from keras.preprocessing.image import load_img, img_to_array, apply_affine_transform
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import scipy
import random
import cv2

class DirectoryDataGenerator(keras.utils.Sequence):
    def __init__(self, image_files, image_labels, face_bbox, augmentor=False, preprocessors=None, batch_size=16, target_sizes=(224,224), shuffle=False, verbose=True, nb_channels=3, zoom=0.0, bb_width=0.5, bb_min=0.0):
        
        self.augmentor = augmentor
        self.preprocessors = preprocessors #should be a function that can be directly called with an image 
        self.batch_size = batch_size
        self.target_sizes = target_sizes
        self.shuffle = shuffle
        self.nb_channels = nb_channels
        
        self.files = image_files	
        self.labels = image_labels
        self.Bbox = face_bbox
        self.nb_files = len(self.files)
        
        self.on_epoch_end() #initialise indexes
        
        self.zoom = zoom
        self.bb_width = bb_width
        self.bb_min = bb_min

        if verbose:
            print('Found {} images.'.format(self.nb_files))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_files / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, Roll, Pitch, Yaw = self.__data_generation(indexes)
        
        #return X, {"Yaw_output": Yaw, "Pitch_output": Pitch, "Roll_output": Roll}
        return X, [Yaw, Pitch, Roll]
        
        #return [X,self.rois], {"region_output": y, "whole_image_output": y, "combine_output": y}
    
    def get_indexes(self):
        return self.indexes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nb_files)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def random_crop(self, image, image_size):
        if image.shape[1]>image_size:
            sz1 = int(image.shape[1]//2)
            sz2 = int(image_size//2)
            #if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            #else:
            # (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
        return image

    def cv2_image_augmentation(self, img, theta=20, tx=10, ty=10, scale=1.):
        
        if scale != 1.:
            scale = np.random.uniform(1-scale, 1+scale)
            
        if theta != 0:
            theta = np.random.uniform(-theta, theta)    
        
        m_inv = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), theta, scale)
        
        
        ''' ADD translation to Rotation Matrix '''
        if tx != 0 or ty != 0:
            tx = np.random.uniform(-tx, tx)
            ty = np.random.uniform(-ty, ty) 
            m_inv[0,2] += tx
            m_inv[1,2] += ty
        
        image = cv2.warpAffine(img, m_inv, (img.shape[1], img.shape[0]), borderMode=1)
        #ones = np.ones(shape=(joints.shape[0], 1))
        #points_ones = np.hstack([joints, ones])
        #joints_gt_aug = m_inv.dot(points_ones.T).T
        return image

    def scipy_image_augmentation(self, img, theta=15, tx=0., ty=0., zoom=0.15):
        
        if zoom != 1:
            #zx, zy = np.random.uniform(1 - zoom, 1 + zoom, 2)
            zx = zy = np.random.uniform(1 - zoom, 1 + zoom)
        else:
            zx, zy = 1, 1
            
        if theta != 0:
            theta = np.random.uniform(-theta, theta)    
        
        #m_inv = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), theta, scale)
        
        
        ''' ADD translation to Rotation Matrix '''
        if tx != 0. or ty != 0.:
            h, w = img.shape[0], img.shape[1]
            ty = np.random.uniform(-ty, ty) * h
            tx = np.random.uniform(-tx, tx) * w

           
        return apply_affine_transform(img, theta=theta, tx=tx, ty=ty, zx=zx, zy=zy)


    def __data_generation(self, indexes):
  
        X = np.empty((self.batch_size, self.target_sizes[0],self.target_sizes[1], self.nb_channels), dtype = K.floatx())
        

        Roll = np.empty((self.batch_size, 1), dtype=K.floatx())#Yaw, Pitch Roll
        Pitch = np.empty((self.batch_size, 1), dtype=K.floatx())
        Yaw = np.empty((self.batch_size, 1), dtype=K.floatx())
        
        # Generate data
        for i in range(len(indexes)):
            #print(i)
            #print(len(indexes))
            img_file = self.files[indexes[i]]
            labels = self.labels[indexes[i]]
            Bbox = self.Bbox[indexes[i]]
            #print(img_file)
            #print(labels[0], labels[1], labels[2])
            #print(Bbox[0],Bbox[1],Bbox[2])
           
            #img_dir = '/home/project' + img_file
            #print(img_dir)
            img = cv2.imread('./Datasets/' + img_file)
            #print(img_file)
            #img = cv2.imread(img_file)
            #print("loaded img: ", img.shape)
            ''' Create a random border width '''
            border_width = 0
            if self.augmentor:
                border_width = np.random.uniform(self.bb_min, self.bb_width) #adjust
            
            x_set = Bbox[0]
            y_set = Bbox[1]
            offset = Bbox[2]
            _offset = int(offset*np.float32(border_width))
            x_start = x_set + offset
            y_start = y_set + offset
            x_end = x_start + offset
            y_end = y_start + offset
            
            '''
            print('offset:', offset, '_offset:',_offset, 'x_set', x_set, 'y_set', y_set)
            print('x_start:', x_start, 'y_start:',y_start, 'x_end', x_end, 'y_end', y_end)
            print('y:', y_start - _offset, 'y2:', y_end + _offset, 'x:', x_start - _offset, 'x1:', x_end + _offset)
            #print(y_set)
            #print(x_end)
            #print(y_end)
            #input()
            '''
        
            src = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[0,0,0])
            #input(src.shape)
            tmp_image = src[y_start - _offset : y_end + _offset, x_start - _offset : x_end + _offset, :]
            tmp_image = cv2.resize(tmp_image, (self.target_sizes[0],self.target_sizes[1]))
            
            ''' Could add data augmentor '''
            if (not self.zoom == 0.0) and self.augmentor:
                tmp_image = self.scipy_image_augmentation(tmp_image, theta=0.0, tx=0., ty=0., zoom=self.zoom)
                tmp_image = cv2.resize(tmp_image, (self.target_sizes[0],self.target_sizes[1]))

            if self.preprocessors: 
                tmp_image = self.preprocessors(tmp_image)
                
            X[i,] = tmp_image
            Roll[i] = labels[0]
            Pitch[i]= labels[1]
            Yaw[i]= labels[2]
            
        return X, Roll, Pitch, Yaw

