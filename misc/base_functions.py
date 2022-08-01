#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:01:30 2020

@author: Hyatt & Ziemann
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.layers import (BatchNormalization,
                                     LayerNormalization,
                                     LeakyReLU,
                                     Lambda,
                                     Convolution2D,
                                     Conv2DTranspose,
                                     Dropout,
                                     Concatenate)

###############################################################################

"""
ResNeXt BLOCK
"""

# This code adapted from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
# with proposed resnext block change from https://arxiv.org/pdf/1603.05027.pdf
# AND with the BN -> ReLU replaced by LeakyReLU -> LN (switch order of norm and activation layers)

def add_common_layers(y,
                      bn=False,
                      ln=False,
                      do=False,
                      bn_axis=-1):

    y = LeakyReLU()(y)
    
    if do:
        y = Dropout(do)(y)
    if bn:
        y = BatchNormalization(axis=bn_axis)(y)
    if ln:
        y = LayerNormalization(axis=bn_axis)(y)

    return y

def grouped_convolution(y,
                        nb_channels,
                        _strides,
                        ksize,
                        dilation,
                        cardinality):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Convolution2D(nb_channels,
                             kernel_size=ksize,
                             strides=_strides,
                             padding='same',
                             dilation_rate=dilation)(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups, and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Convolution2D(_d,
                                    kernel_size=ksize,
                                    strides=_strides,
                                    padding='same',
                                    dilation_rate=dilation)(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)

    return y

# Adapted the grouped_convolution function above to allow for upsampling
def grouped_transpose_convolution(y,
                                  nb_channels,
                                  _strides,
                                  ksize,
                                  cardinality):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv2DTranspose(nb_channels,
                               kernel_size=ksize,
                               strides=_strides,
                               padding='same')(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups, and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Conv2DTranspose(_d,
                                      kernel_size=ksize,
                                      strides=_strides,
                                      padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)

    return y

# nb_channels_in IS THE REDUCED NUMBER OF CHANNELS IN THE BOTTLENECK/DIMENSION REDUCTION PHASE.
# nb_channels_out IS THE NUMBER OF CHANNELS IN THE BLOCK OUTPUT.
def residual_block(y,
                   nb_channels_in,
                   nb_channels_out,
                   _strides=(1, 1),
                   _project_shortcut=False,
                   ksize=(4, 4),
                   cardinality=4,
                   bn=False,
                   ln=False,
                   do=False,
                   bn_axis=-1):

    """
    Our network consists of a stack of residual blocks. These blocks have the same topology, and are subject to two simple rules:

    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """

    # This is the identity / shortcut path
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)
    
    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)

    y = grouped_convolution(y,
                            nb_channels_in,
                            _strides=_strides,
                            ksize=ksize,
                            dilation=(1,1),
                            cardinality=cardinality)

    # Map the aggregated branches to desired number of output channels
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)

    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # Add to the shortcut
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut \
    or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Convolution2D(nb_channels_out,
                                 kernel_size=(1, 1),
                                 strides=_strides,
                                 padding='same')(shortcut)

    # Add the shortcut and the transformation block
    y = layers.add([shortcut, y])

    return y

# Same as residual_block, but uses 4 parallel dilated convolutions. Must have strides 1,1.
def dilated_residual_block(y,
                           nb_channels_in,
                           nb_channels_out,
                           _strides=(1, 1),
                           _project_shortcut=False,
                           ksize=(4, 4),
                           cardinality=4,
                           bn=False,
                           ln=False,
                           do=False,
                           bn_axis=-1):

    """
    This block uses the same structure as a residual block, but splits the input into 4 parallel dilated convolutions of
    different dilation amounts (currently 1x, 2x, 3x, 4x). This is done to increase the effective distance that these
    convolutions activate over on the input image, to better predict long-range effects.
    """

    # This is the identity / shortcut path
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)
    
    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)

    # Branch into 4 parallel convolutions, each with a different dilation value. Note, currently requires 1,1 stride
    parallel_1 = grouped_convolution(y,
                                     nb_channels_in,
                                     _strides=_strides,
                                     ksize=ksize,
                                     dilation=(1, 1),
                                     cardinality=cardinality)
    
    parallel_2 = grouped_convolution(y,
                                     nb_channels_in,
                                     _strides=_strides,
                                     ksize=ksize,
                                     dilation=(2, 2),
                                     cardinality=cardinality)
    
    parallel_3 = grouped_convolution(y,
                                     nb_channels_in,
                                     _strides=_strides,
                                     ksize=ksize,
                                     dilation=(3, 3),
                                     cardinality=cardinality)
    
    parallel_4 = grouped_convolution(y,
                                     nb_channels_in,
                                     _strides=_strides,
                                     ksize=ksize,
                                     dilation=(4, 4),
                                     cardinality=cardinality)
    
    # Concatenate parallel branches
    y = Concatenate(axis=-1)([parallel_1,
                              parallel_2,
                              parallel_3,
                              parallel_4])
    
    # Map the aggregated branches to desired number of output channels
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)

    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # Add to the shortcut
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut \
    or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Convolution2D(nb_channels_out,
                                 kernel_size=(1, 1),
                                 strides=_strides,
                                 padding='same')(shortcut)

    # Add the shortcut and the transformation block
    y = layers.add([shortcut, y])

    return y

# Adapted the residual_block function above to allow for upsampling
# nb_channels_in IS THE REDUCED NUMBER OF CHANNELS IN THE BOTTLENECK/DIMENSION REDUCTION PHASE.
# nb_channels_out IS THE NUMBER OF CHANNELS IN THE BLOCK OUTPUT.
def transpose_residual_block(y,
                             nb_channels_in,
                             nb_channels_out,
                             _strides=(1, 1),
                             _project_shortcut=False,
                             ksize=(4, 4),
                             cardinality=4,
                             bn=False,
                             ln=False,
                             do=False,
                             bn_axis=-1):

    """
    These blocks are similar to residual blocks (see above), and are subject to two simple rules:

    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is up-sampled by a factor of 2, the width of the blocks is divided by a factor of 2.
    """

    # This is the identity / shortcut path
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)
    
    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)
    
    # Branch into 4 parallel transpose convolutions, each with a different dilation value   
    y = grouped_transpose_convolution(y,
                                      nb_channels_in,
                                      _strides=_strides,
                                      ksize=ksize,
                                      cardinality=cardinality)
    
    # Map the aggregated branches to desired number of output channels
    y = add_common_layers(y,
                          bn,
                          ln,
                          do,
                          bn_axis)

    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # Add to the shortcut
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut \
    or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2DTranspose(nb_channels_out,
                                   kernel_size=(1, 1),
                                   strides=_strides,
                                   padding='same')(shortcut)

    # Add the shortcut and the transformation block
    y = layers.add([shortcut, y])

    return y

def tanh2(x, name = None):
    """
    Custom activation function to deal with residuals. Datasets are normalized [-1,1]; residuals can be in range [-2,2].
    Rescales tanh activation to [-2,2]. Uses (x/2) to retain same gradient behavior around zero as tanh().
    
    2*tanh(x/2)
    """
    y = tf.math.multiply(2.0, tf.math.tanh(tf.math.divide(x, 2.0, name), name), name)
    return y
    

###############################################################################

"""
FUNCTIONS FOR LOADING AND PREPROCESSING RANDOMVPLDATASET NUMPY ARRAYS INTO TENSORFLOW DATASETS
"""

@tf.function
def rotate(input_tensor):

    """
    Rotation augmentation
    
    Args:
        input_tensor: Image to rotate

    Returns:
        Rotated image
    """

    # Rotate 0, 90, 180, 270 degrees. Upper value (maxval=4) is excluded.
    return tf.image.rot90(input_tensor, 
                          tf.random.uniform(shape=[], 
                                            minval=0, 
                                            maxval=4, 
                                            dtype=tf.int32))


@tf.function
def flip(input_tensor):

    """
    Flip augmentation

    Args:
        input_tensor: Image to flip

    Returns:
        Randomly flipped image
    """

    input_tensor = tf.image.random_flip_left_right(input_tensor)
    input_tensor = tf.image.random_flip_up_down(input_tensor)

    return input_tensor


@tf.function
def dtype_set(x):
    
    """
    Sets the Dataset dtype

    Args:
        x: Input dataset.

    Returns:
        x: dtype float32
    """
    
    x = tf.cast(x, tf.float32)
    return x


def make_dataset(datapath_0,
                 datapath_1,
                 residual,
                 augment,
                 batch_size,
                 shuffle):

    """
    Tensorflow input pipeline.

    This takes in RandomVPLDataset Numpy Arrays (either two reflection outputs if using residuals, or a building input and a 
    reflection target) and loads them into a two-channel Tensorflow Dataset (first channel for the model input, either a 
    building location or previous reflection step; second channel for the target, either 0 reflection step or residual). If
    using a residual, set residual=True, otherwise set residual=False.
                                                                             
    This utilizes dataset augmentation (flipping/rotating), and shuffles the data. For validation data, you'll want to set
    augment=False and shuffle=False. 
                                                                               
    XXX NOTE: This does not take previous model predictions to calculate residuals or for input. It uses the previous reflection
    step's VPL output. This will need to be changed once the models are fine tuned and you're ready to chain neural nets.
    """

    def load_inputs(datapath_0, 
                    datapath_1,
                    residual):

        """
        Args:
            datapath_0: The filepath to either the building input numpy array (if training first model), or the previous VPL 
            reflection step numpy array.
            datapath_1: The filepath to the target VPL reflection data numpy array.
            residual: Boolean. Set to True to set the target data to the residual of (datapath_1 array - datapath_0 array)

        Returns:
            combined_dataset: A numpy array containing input (channel 0) and target (channel 1) data as a 2 channel array.
        """

        # Load numpy arrays. EITHER loads previous reflection case as input, and calculates residual, OR loads bdg input as input
        input_dataset = np.load(datapath_0)
        target_dataset = np.load(datapath_1)
        
        # Calculate the residual, and set that as the target.
        if residual:
            target_dataset = target_dataset - input_dataset
        
        # Reshape files to (<>,<>, <>, 1) if they lost 4th dimension in the .npy transfer (tends to happen)
        input_dataset.shape += (1,) * (4 - input_dataset.ndim)
        target_dataset.shape += (1,) * (4 - target_dataset.ndim)
        
        # Concatenate the input and target data into a single 2-channel dataset
        combined_dataset = np.concatenate((input_dataset, 
                                           target_dataset),
                                          axis=3)
        data_size = input_dataset.shape[0] 
        
        return combined_dataset, data_size
       
    @tf.function
    def uncombine_dataset(combined_image):
        
        """
        Separates input's channels
    
        Args:
            combined_image: 2 channel image
    
        Returns:
            Tuple: (image_channel_0, image_channel_1)
        """
        
        input_image = tf.expand_dims(combined_image[...,0],
                                     axis=-1)
        output_image = tf.expand_dims(combined_image[...,1],
                                      axis=-1)
        
        return (input_image, output_image)
    

        
    
    #-- Load input and target numpy arrays into a 2 channel array. Will calculate a residual if required.
    combined_dataset, data_size = load_inputs(datapath_0,
                                              datapath_1,
                                              residual)
    
    #-- Convert numpy arrays into Dataset
    combined_dataset = tf.data.Dataset.from_tensor_slices(combined_dataset)
    
    #-- Set the Dataset dtype to float32
    combined_dataset = combined_dataset.map(dtype_set,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #-- Cache the dataset for faster recall
    combined_dataset = combined_dataset.cache()
    
    #-- Shuffle the dataset if requested (note: shuffles both channels together, to keep inputs/targets together)
    if shuffle:
        shuffle_buffer_size = 2*data_size # Set the shuffle buffer to twice the size of the dataset.
        combined_dataset = combined_dataset.shuffle(shuffle_buffer_size)
        
    #-- Batch the dataset
    combined_dataset = combined_dataset.batch(batch_size)
    
    #-- Augment the dataset, if desired, with random rotations and flips. ~8x more effective samples.
    if augment:
        # Note: We tried batching and unbatching here, and it was slower to batch by ~30%
        combined_dataset = combined_dataset.map(rotate,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        combined_dataset = combined_dataset.map(flip,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #-- Separate the input and target channels for use in the neural net
    combined_dataset = combined_dataset.map(uncombine_dataset,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #-- Prefetch dataset to save computational time.
    combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return combined_dataset


def make_dataset_mInput(bdg_datapath,
                        n2_datapath,
                        n1_datapath,
                        n_datapath,
                        augment,
                        batch_size,
                        shuffle):

    """
    Tensorflow input pipeline.

    Similar to make_dataset, but accepts bdg, n-2, and n-1 reflection data as input data, and n-reflection data as the target.
    
    Prepares a tf dataset with ((bdg, n2, n1), n) for multi input training. 
    """

    def load_inputs(bdg_datapath,
                    n2_datapath,
                    n1_datapath,
                    n_datapath):

        """
        Args:
            bdg_data: Building location data (0's and 1's). Input 1
            n2_data: n-2 reflection data. For example, if predicting 2 reflection, this is 0 reflection. Input 2
            n1_data: n-1 reflection data. For example, if predicting 2 reflection, this is 1 reflection. Input 3
            n_data: n-reflection data. This is the data being predicted. Target 1

        Returns:
            combined_dataset: A numpy array containing ((bdg, n2, n1), n).
        """

        bdg_data = np.load(bdg_datapath)
        n2_data = np.load(n2_datapath)
        n1_data = np.load(n1_datapath)
        n_data = np.load(n_datapath)
        
        # Calculate the n residual from n - n1, and set that as the target. This forces the model to learn exclusively the n
        # reflections, and no reflections from previous steps
        target_data = n_data - n1_data
        
        # TODO: When trying 3-reflection and above, implement the (n-2) residual. Skip it for now, b/c it doesn't make sense for
        # the zero reflection case.
        # n2_data_res = n2_data - n3_data  # Note: would need to import n-3 reflection data to create this.
        
        # Calculate the n-1 residual and use that as the (n-1) data. 
        n1_data = n1_data - n2_data
        
        # Reshape files to (<>, <>, <>, 1) if they lost 4th dimension in the .npy transfer (tends to happen)
        bdg_data.shape += (1,) * (4 - bdg_data.ndim)
        n2_data.shape += (1,) * (4 - n2_data.ndim)
        n1_data.shape += (1,) * (4 - n1_data.ndim)
        target_data.shape += (1,) * (4 - target_data.ndim)
        
        # Concatenate the data into a single 4-channel dataset for transformations. This will be de-coupled later for training.
        combined_dataset = np.concatenate((bdg_data, 
                                           n2_data,
                                           n1_data,
                                           target_data),
                                          axis=3)
        # Grab the number of cities in the dataset for use in shuffle buffer
        data_size = bdg_data.shape[0] 
        
        return combined_dataset, data_size
       
    @tf.function
    def uncombine_dataset(combined_image):
        
        """
        Separates input's channels
    
        Args:
            combined_image: 4 channel image
    
        Returns:
            Tuple: ((bdg_input, n2_input, n1_input), n_target)
        """
        
        bdg_input = tf.expand_dims(combined_image[...,0],
                                   axis=-1)
        n2_input = tf.expand_dims(combined_image[...,1],
                                  axis=-1)
        n1_input = tf.expand_dims(combined_image[...,2],
                                  axis=-1)
        n_target = tf.expand_dims(combined_image[...,3],
                                  axis=-1)
        
        return ((bdg_input, n2_input, n1_input), n_target)
    

        
    
    #-- Load input and target numpy arrays into a 4 channel array.
    combined_dataset, data_size = load_inputs(bdg_datapath,
                                              n2_datapath,
                                              n1_datapath,
                                              n_datapath)
    
    #-- Convert numpy arrays into Dataset
    combined_dataset = tf.data.Dataset.from_tensor_slices(combined_dataset)
    
    #-- Set the Dataset dtype to float32
    combined_dataset = combined_dataset.map(dtype_set,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #-- Cache the dataset for faster recall
    combined_dataset = combined_dataset.cache()
    
    #-- Shuffle the dataset if requested (note: shuffles all channels together, to keep inputs/targets together)
    if shuffle:
        shuffle_buffer_size = 2*data_size # Set the shuffle buffer to twice the size of the dataset.
        combined_dataset = combined_dataset.shuffle(shuffle_buffer_size)
        
    #-- Batch the dataset
    combined_dataset = combined_dataset.batch(batch_size)
    
    #-- Augment the dataset, if desired, with random rotations and flips. ~8x more effective samples.
    if augment:
        # Note: We tried batching and unbatching here, and it was slower to batch by ~30%
        combined_dataset = combined_dataset.map(rotate,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        combined_dataset = combined_dataset.map(flip,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #-- Separate the input and target channels for use in the neural net
    combined_dataset = combined_dataset.map(uncombine_dataset,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #-- Prefetch dataset to save computational time.
    combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return combined_dataset
