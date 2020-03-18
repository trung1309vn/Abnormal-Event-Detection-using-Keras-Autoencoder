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

class Autoencoder_MultiFilter():
    def __init__(self, img_row, img_col, img_len, channel):
        # Input shape
        self.img_rows = img_row
        self.img_cols = img_col
        self.channels = channel
        self.seq_len = img_len
        self.img_shape = ( self.seq_len, self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(0.0002, 0.5)

        # Frame to Optical flow
        self.autoencoder, self.encoder = self.build_autoencoder() #, self.encoder_late
        
        # The combined model for Image to Optical flow
        self.autoencoder.compile(optimizer='nadam', loss='mse')
        
    def build_autoencoder(self):
        #---------------------------------------------------------------------------#
        #Encoder---------------------------------------------------------------#
        #---------------------------------------------------------------------------#
        inp_seq = Input(shape=self.img_shape)
        
       
        # First Encoder layer
        enc_1_1 = Conv3D(filters=64,kernel_size=(1,11,11),strides=(1,1,1),
                         padding='same')(inp_seq)
        enc_1_2 = Conv3D(filters=64,kernel_size=(1,7,7),strides=(1,1,1),
                         padding='same')(inp_seq)
        enc_1_3 = Conv3D(filters=64,kernel_size=(1,3,3),strides=(1,1,1),
                         padding='same')(inp_seq)
        
        enc_1 = concatenate([enc_1_1, enc_1_2, enc_1_3])
        enc_1 = Conv3D(filters=128,kernel_size=(1,11,11),strides=(1,4,4),
                         padding='valid')(enc_1)
        enc_1 = BatchNormalization(momentum=0.8)(enc_1)
        enc_1 = LeakyReLU(0.2)(enc_1)
        
        # Second Encoder layer
        enc_2_1 = Conv3D(filters=32,kernel_size=(1,7,7),strides=(1,1,1),
                         padding='same')(enc_1)
        enc_2_2 = Conv3D(filters=32,kernel_size=(1,5,5),strides=(1,1,1),
                         padding='same')(enc_1)
        enc_2_3 = Conv3D(filters=32,kernel_size=(1,3,3),strides=(1,1,1),
                         padding='same')(enc_1)
        
        enc_2 = concatenate([enc_2_1, enc_2_2, enc_2_3])
        enc_2 = Conv3D(filters=64,kernel_size=(1,5,5),strides=(1,2,2),
                         padding='valid')(enc_2)
        enc_2 = BatchNormalization(momentum=0.8)(enc_2)
        enc_2 = LeakyReLU(0.2)(enc_2)
        
        # Third Encoder layer
        enc_3_1 = Conv3D(filters=16,kernel_size=(1,7,7),strides=(1,1,1),
                         padding='same')(enc_2)
        enc_3_2 = Conv3D(filters=16,kernel_size=(1,5,5),strides=(1,1,1),
                         padding='same')(enc_2)
        enc_3_3 = Conv3D(filters=16,kernel_size=(1,3,3),strides=(1,1,1),
                         padding='same')(enc_2)
        
        enc_3 = concatenate([enc_3_1, enc_3_2, enc_3_3])
        enc_3 = Conv3D(filters=32,kernel_size=(1,5,5),strides=(1,3,3),
                         padding='valid')(enc_3)
        enc_3 = BatchNormalization(momentum=0.8)(enc_3)
        enc_3 = LeakyReLU(0.2)(enc_3)

        # Bottle neck LSTM
        lstm_1 = ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,
            padding='same',return_sequences=True, data_format='channels_last')(enc_3)
        
        lstm_2 = ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,
            padding='same',return_sequences=True, data_format='channels_last')(lstm_1)
         
        # First Decoder layer
        dec_1 = Conv3DTranspose(filters=64,kernel_size=(1,5,5),strides=(1,3,3),
                                 padding='valid')(lstm_2)
        dec_1 = BatchNormalization(momentum=0.8)(dec_1)
        dec_1 = LeakyReLU(0.2)(dec_1)
        
        # First Decoder layer
        dec_2 = Conv3DTranspose(filters=128,kernel_size=(1,5,5),strides=(1,2,2),
                                 padding='valid')(dec_1)
        dec_2 = BatchNormalization(momentum=0.8)(dec_2)
        dec_2 = LeakyReLU(0.2)(dec_2)
 
        # Second Decoder layer
        dec_3 = Conv3DTranspose(filters=1,kernel_size=(1,11,11),strides=(1,4,4),
                                 padding='valid', activation='sigmoid')(dec_2)
        
        autoencoder = Model(inp_seq, dec_3)
        encoder = Model(inp_seq, lstm_1)
        
        print('Autoencoder model summary')
        autoencoder.summary()
        return (autoencoder, encoder)

    def train(self, epochs, X_train, X_val, batch_size=1):
        
        print("Training model")
        prev_loss = self.autoencoder.evaluate(X_train, X_train, verbose=1, batch_size=batch_size)
        print(prev_loss)
        past = datetime.now()
        
        for epoch in range(epochs):

            # Get number of iteration            
            # Loss initial of combine, image generator, discriminator
            
            autoencoder_losses = []
                   
            # Training section
            autoencoder_history = self.autoencoder.fit(x=X_train, y=X_train, 
                        validation_data=(X_val, X_val), epochs=1, batch_size=batch_size, verbose=1)
            autoencoder_losses.append(autoencoder_history.history["val_loss"])

            #d_count += 1
            #tmp_d_l = d_l[0]
            
            now = datetime.now()
            print("End epoch")
            print("\nEpoch {}/{} - {:.1f}s".format(epoch, epochs, (now - past).total_seconds()))
            print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
            
            self.save_imgs(epoch)
            if (prev_loss > np.mean(autoencoder_losses)):
                prev_loss = np.mean(autoencoder_losses)
                self.save_model()
            past = now
                
            #'''
    def save_model(self):
        enc = 'Saved_model/Autoencoder_Ped1_MultiFilter_enhanced.sav'
        self.autoencoder.save(enc)

    def load_model(self):
        enc = 'Saved_model/Autoencoder_Ped1_MultiFilter_enhanced.sav'
        self.autoencoder.load_weights(enc)

    def save_imgs(self, epoch, X_test):
        r, c = 5, 2
        # Select a random half of images
        idx = np.random.randint(0, X_test.shape[0] - 1, 1)
        imgs = X_test[idx]
        
        print("Save test images")
        gen_imgs = self.autoencoder.predict(imgs, batch_size=1, verbose=1)
        
        fig=plt.figure(figsize=(30, 30))
        for i in range(0, r):
            fig.add_subplot(r, c, i*2 + 1)
            plt.imshow(imgs[0,i,:,:,0], cmap='gray')
            fig.add_subplot(r, c, i*2 + 2)
            plt.imshow(gen_imgs[0,i,:,:,0], cmap='gray')

        plt.savefig("images/img_opt_%d.png" %epoch)
        plt.close()
