from keras.utils import Sequence
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from keras.layers import BatchNormalization, Activation, ConvLSTM2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import preprocessor

import numpy as np

class Dual_Autoencoder_MultiFilter():
    def __init__(self, img_row, img_col, img_len, channel):
        # Input shape
        self.img_rows = img_row
        self.img_cols = img_col
        self.channels = channel
        self.seq_len = img_len
        self.img_shape = (self.seq_len, self.img_rows, self.img_cols, self.channels)
        self.opflow_shape = (self.seq_len, self.img_rows, self.img_cols, 2)
        
        optimizer = Adam(0.0002, 0.5)

        # Frame to Optical flow
        self.autoencoder, self.encoder = self.build_autoencoder(1, 2) #, self.encoder_late
        
        # Optical flow to Frame
        self.auto_opflow, self.encoder_opflow = self.build_autoencoder(2, 1) #, self.encoder_late
        
        # The combined model for Image to Optical flow
        self.autoencoder.compile(optimizer='nadam', loss='mse')
        self.auto_opflow.compile(optimizer='nadam', loss='mse')
        
    def build_autoencoder(self, in_channels, out_channels):
        #---------------------------------------------------------------------------#
        #Encoder---------------------------------------------------------------#
        #---------------------------------------------------------------------------#
        inp_seq = Input(shape=(self.seq_len, self.img_rows, self.img_cols, in_channels))
        
        # First Encoder layer
        enc_1_1 = Conv3D(filters=64,kernel_size=(1,11,11),strides=(1,1,1),
                         padding='same', activation='relu')(inp_seq)
        enc_1_2 = Conv3D(filters=64,kernel_size=(1,7,7),strides=(1,1,1),
                         padding='same', activation='relu')(inp_seq)
        enc_1_3 = Conv3D(filters=64,kernel_size=(1,3,3),strides=(1,1,1),
                         padding='same', activation='relu')(inp_seq)
        
        enc_1 = concatenate([enc_1_1, enc_1_2, enc_1_3])
        enc_1 = Conv3D(filters=64*3,kernel_size=(1,11,11),strides=(1,4,4),
                         padding='valid')(enc_1)
        enc_1 = BatchNormalization(momentum=0.8)(enc_1)
        enc_1 = LeakyReLU(0.2)(enc_1)
        
        # Second Encoder layer
        enc_2_1 = Conv3D(filters=32,kernel_size=(1,7,7),strides=(1,1,1),
                         padding='same', activation='relu')(enc_1)
        enc_2_2 = Conv3D(filters=32,kernel_size=(1,5,5),strides=(1,1,1),
                         padding='same', activation='relu')(enc_1)
        enc_2_3 = Conv3D(filters=32,kernel_size=(1,3,3),strides=(1,1,1),
                         padding='same', activation='relu')(enc_1)
        
        enc_2 = concatenate([enc_2_1, enc_2_2, enc_2_3])
        enc_2 = Conv3D(filters=32*3,kernel_size=(1,5,5),strides=(1,2,2),
                         padding='valid')(enc_2)
        enc_2 = BatchNormalization(momentum=0.8)(enc_2)
        enc_2 = LeakyReLU(0.2)(enc_2)
        
        # Bottle neck LSTM        
        lstm = ConvLSTM2D(filters=32*3,kernel_size=(3,3),strides=1,
                padding='same',return_sequences=True, data_format='channels_last')(enc_2)
        
        # First Decoder layer
        dec_1 = Conv3DTranspose(filters=64*3,kernel_size=(1,5,5),strides=(1,2,2),
                                 padding='valid')(lstm)
        dec_1 = BatchNormalization(momentum=0.8)(dec_1)
        dec_1 = LeakyReLU(0.2)(dec_1)
        
        # Second Decoder layer
        dec_2 = Conv3DTranspose(filters=out_channels,kernel_size=(1,11,11),strides=(1,4,4),
                                 padding='valid', activation='sigmoid')(dec_1)
        
        autoencoder = Model(inp_seq, dec_2)
        encoder = Model(inp_seq, lstm)
        
        if (in_channels == 1):
            print('Autoencoder model summary')
            autoencoder.summary()
        else:
            print('Auto_opflow model summary')
            autoencoder.summary()
            
        return (autoencoder, encoder)

    def train(self, epochs, X_train, X_val, batch_size=1):
        
        print("Training model")
        past = datetime.now()
        
        for epoch in range(epochs):

            # Get number of iteration            
            # Loss initial of combine, image generator, discriminator
            
            autoencoder_losses = []
            auto_opflow_losses = []
                   
            # Training section
            autoencoder_history = self.autoencoder.fit(x=X_train, y=X_train_opflow,
                                    epochs=1, batch_size=batch_size, verbose=1)
            autoencoder_losses.append(autoencoder_history.history["loss"])
            auto_opflow_history = self.auto_opflow.fit(x=X_train_opflow, y=X_train,
                                    epochs=1, batch_size=batch_size, verbose=1)
            auto_opflow_losses.append(auto_opflow_history.history["loss"])

            #d_count += 1
            #tmp_d_l = d_l[0]
            
            now = datetime.now()
            print("End epoch")
            print("\nEpoch {}/{} - {:.1f}s".format(epoch, epochs, (now - past).total_seconds()))
            print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
            print("Auto Opflow Loss: {}".format(np.mean(auto_opflow_losses)))
            
            self.save_imgs(epoch)
            past = now
                
            #'''
    def save_model(self):
        enc = 'Saved_model/DualAuto_I2O_Ped2_MultiFilter_enhanced.sav'
        
        self.autoencoder.save(enc)
        
        enc_op = 'Saved_model/DualAuto_O2I_Ped2_MultiFilter_enhanced.sav'
        
        self.auto_opflow.save(enc_op)


    def load_model(self):
        enc = 'Saved_model/DualAuto_I2O_Ped2_MultiFilter_enhanced.sav'
        
        self.autoencoder.load_weights(enc)
        
        enc_op = 'Saved_model/DualAuto_O2I_Ped2_MultiFilter_enhanced.sav'
        
        self.auto_opflow.load_weights(enc_op)
        
    def show_optical_flow(self, flow):
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(np.array(hsv, dtype=np.uint8),cv2.COLOR_HSV2BGR)
        plt.imshow(bgr)

    def save_imgs(self, epoch):
        r, c = 5, 4
        # Select a random half of images
        idx = np.random.randint(0, X_test.shape[0] - 1, 1)
        imgs = X_test[idx]
        opts = X_test_opflow[idx]
        
        print("Save test images")
        gen_opts = self.autoencoder.predict(imgs, batch_size=1, verbose=1)
        gen_imgs = self.auto_opflow.predict(opts, batch_size=1, verbose=1)
        
        fig=plt.figure(figsize=(30, 30))
        for i in range(0, r):
            fig.add_subplot(r, c, i*4 + 1)
            plt.imshow(imgs[0,i,:,:,0], cmap='gray')
            fig.add_subplot(r, c, i*4 + 3)
            show_optical_flow(opts[0,i,:,:])
            fig.add_subplot(r, c, i*4 + 2)
            plt.imshow(gen_imgs[0,i,:,:,0], cmap='gray')
            fig.add_subplot(r, c, i*4 + 4)
            show_optical_flow(gen_opts[0,i,:,:])

        plt.savefig("images/img_opt_%d.png" %epoch)
        plt.close()

