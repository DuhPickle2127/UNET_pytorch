#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:04:48 2020

@author: Hyatt & Ziemann
"""

import tensorflow as tf

from tensorflow.keras.models import (Model)

from tensorflow.keras.layers import (Activation,
                                     Input,
                                     Convolution2D,
                                     Concatenate)

from base_functions import (residual_block,
                            dilated_residual_block,
                            transpose_residual_block,
                            tanh2)

def make_codec_model(x_shape,
                     LAYER_NORM,
                     DROPOUT,
                     TESTING): # XXX NOT USED. TODO: IMPLEMENT

    '''
    This creates our codec model. This is a modified U-NET style autoencoder that encodes the image down w/ 4 downsamples, then
    reconstructs the input (decoding) with 4 upsamples. Each upsample is followed by a skip connection, then a dilated 
    convolution. Various depth layers between each up/down-sampling step.
    '''
    
    x_height = x_shape[0] # height of target reconstruction
    x_width = x_shape[1] # width of target reconstruction
    x_depth = x_shape[2] # depth of target reconstruction
    
    # (i) Sparse building file input
    build_branch_input = Input(shape=(x_height,
                                      x_width,
                                      x_depth),
                               name='C_input_build_branch')

    # Transform the building file input by itself.  This has 2 effects:
    # 1) the conv layer here should effectively learn an optimal preprocessing transformation, and
    # 2) it moves this input into a higher number of dimensions, in this case 96 (but can be whatever you want).
    #    This gives the network leeway to learn complex transformations.
    build_branch = Convolution2D(filters=96,
                                 kernel_size=4,
                                 strides=(1,1),
                                 padding='same',
                                 kernel_initializer='he_normal')(
                                     build_branch_input)

    ###################################################################
    ###################################################################
    ###################################################################

    #-- Currently unused. Placeholder for inclusion of terrain data.
    #-- Similar input can be used for antenna location or similar data.
    
    # # (ii) Terrain file input.
    # terr_branch_input = Input(shape=(grid_height,
    #                                  grid_width,
    #                                  grid_depth),
    #                           name='E_input_terr_branch')

    # # Transform the terrain file input by itself.
    # terr_branch = conv(terr_branch_input,
    #                    n_filters=32, # adjust as needed, started at 32
    #                    kernel_size=4)  # adjust as needed

    ###################################################################
    ###################################################################
    ###################################################################

    #-- Currently unused.
    
    # Concatenate the two inputs.  This is the "real" input to your encoder.
    # model_0 = Concatenate(axis=-1)([build_branch,
    #                              terr_branch])

    # Now for some stacked convolution blocks.  These will gradually decrease the number of channels AND the resolution
    # of the image (NOTE: No longer reduces channels).  We want to get both of these small enough so that we can flatten 
    # the output of the last conv layer, feed it into a dense layer or two, and not have eighty trillion parameters in 
    # that dense layer.  This requires decreasing both.

    ### BLOCK 0 #######################################################
    # Pull skip connection 1. #########################################
    ###################################################################

    skip_1 = build_branch  # TODO Was build_branch_input, changed to build_branch. Test this.
    
    ### BLOCK 1 #######################################################
    # Apply dilated convolution
    # Uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################

    model = dilated_residual_block(build_branch,
                                   nb_channels_in=96,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 2 #######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    ### BLOCK 3 #######################################################
    # Decrease the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           _strides=(2,2),
                           cardinality=8,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 4 #######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 5 #######################################################
    # Pull skip connection 2 ##########################################
    ###################################################################

    skip_2 = model
    
    ### BLOCK 6 #######################################################
    # Apply dilated convolution
    # Uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################

    model = dilated_residual_block(model,
                                   nb_channels_in=96,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 7 #######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 8 #######################################################
    # Decrease the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           _strides=(2,2),
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 9 #######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    

    ### BLOCK 10 ######################################################
    # Pull skip connection 3 ##########################################
    ###################################################################

    skip_3 = model
    
    ### BLOCK 11 ######################################################
    # Apply dilated convolution
    # Uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################

    model = dilated_residual_block(model,
                                   nb_channels_in=96,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 12 ######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    ### BLOCK 13 ######################################################
    # Decrease the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           _strides=(2,2),
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 14 ######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 15 ######################################################
    # Pull skip connection 4 ##########################################
    ###################################################################

    skip_4 = model

    ### BLOCK 16 ######################################################
    # Decrease the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           _strides=(2,2),
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 17 ######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    #==================================================================
    # End of encoding. Begin decoder. =================================
    #==================================================================
    
    
    ### BLOCK 0 #######################################################
    # Reduce channel size of skip connections. ########################
    ###################################################################
    
    skip_1_small = Convolution2D(32,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same')(skip_1)
    
    skip_2_small = Convolution2D(32,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same')(skip_2)
    
    skip_3_small = Convolution2D(32,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same')(skip_3)
    
    skip_4_small = Convolution2D(32,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same')(skip_4)

    ### BLOCK 1 #######################################################
    # Increase the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = transpose_residual_block(model,
                                     nb_channels_in=96,
                                     nb_channels_out=96,
                                     _strides=(2,2),
                                     cardinality=12,
                                     do=DROPOUT,
                                     ln=LAYER_NORM)

    ### BLOCK 2 #######################################################
    # Bring in fourth skip connection. ################################
    ###################################################################

    model = Concatenate(axis=-1)([model,
                                  skip_4_small])

    ### BLOCK 3 #######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=128,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=16,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 4 #######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 5 #######################################################
    # Increase the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = transpose_residual_block(model,
                                     nb_channels_in=96,
                                     nb_channels_out=96,
                                     _strides=(2,2),
                                     cardinality=12,
                                     do=DROPOUT,
                                     ln=LAYER_NORM)
    
    ### BLOCK 6 #######################################################
    # Bring in third skip connection. #################################
    ###################################################################

    model = Concatenate(axis=-1)([model,
                                  skip_3_small])
    
    ### BLOCK 7 #######################################################
    # Apply dilated convolution, which uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################

    model = dilated_residual_block(model,
                                   nb_channels_in=128,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 8 #######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 9 #######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 10 ######################################################
    # Increase the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = transpose_residual_block(model,
                                     nb_channels_in=96,
                                     nb_channels_out=96,
                                     _strides=(2,2),
                                     cardinality=12,
                                     do=DROPOUT,
                                     ln=LAYER_NORM)
    
    ### BLOCK 11 ######################################################
    # Bring in second (reduced) skip connection
    ###################################################################
    
    model = Concatenate(axis=-1)([model,
                                  skip_2_small])
    
    ### BLOCK 12 ######################################################
    # Apply dilated convolution, which uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################

    model = dilated_residual_block(model,
                                   nb_channels_in=128,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 13 ######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 14 ######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 15 ######################################################
    # Increase the resolution.  Do NOT decrease the number of channels.
    ###################################################################

    model = transpose_residual_block(model,
                                     nb_channels_in=96,
                                     nb_channels_out=96,
                                     _strides=(2,2),
                                     cardinality=12,
                                     do=DROPOUT,
                                     ln=LAYER_NORM)
    
    ### BLOCK 16 ######################################################
    # Bring in first (reduced) skip connection ########################
    ###################################################################
    
    model = Concatenate(axis=-1)([model,
                                  skip_1_small])
    
    ### BLOCK 17 ######################################################
    # Apply dilated convolution, which uses 4 parallel convolutions dilated at (1,1) ; (2,2) ; (3,3) ; and (4,4)
    ###################################################################
                                  
    model = dilated_residual_block(model,
                                   nb_channels_in=128,
                                   nb_channels_out=256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   do=DROPOUT,
                                   ln=LAYER_NORM)

    ### BLOCK 18 ######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=256,
                           nb_channels_out=96,
                           _project_shortcut=True,
                           cardinality=32,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 19 ######################################################
    # For depth (no change in resolution/number of channels) ##########
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=96,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)

    ### BLOCK 20 ######################################################
    # Decrease the number of channels.  Do NOT decrease the resolution.
    ###################################################################

    model = residual_block(model,
                           nb_channels_in=96,
                           nb_channels_out=1,
                           _project_shortcut=True,
                           cardinality=12,
                           do=DROPOUT,
                           ln=LAYER_NORM)
    
    model = Activation(activation=tanh2)(model)

    receiver = model # final shape (grid_height, grid_width, 1)

    C = Model([build_branch_input],  # input (i)
              [receiver], 	     # output (I)
              name='codec')
        
    return C

