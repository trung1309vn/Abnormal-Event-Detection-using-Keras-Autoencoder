import cv2
import os
import pims
import random
import matplotlib.pyplot as plt
import numpy as np

img_row = 227
img_col = 227
channel = 1
img_len = 5

def normalize_image(seq):
    for img_idx in range(len(seq)):
        seq[img_idx] = cv2.resize(seq[img_idx], (img_row,img_col))
        seq[img_idx] = (seq[img_idx]) / 255.0
    return seq

def count_set(path):
    list_dir = []
    for data in os.listdir(path):
        if ('.' not in data):
            list_dir.append(data)
    return len(list_dir)

def read_image_sequence(path, ext='.tif'):
    image_arr = []
    img_list = os.listdir(path)
    for img in img_list:
        if (ext in img):
            image_arr.append(plt.imread(path + '/' + img, ext))
    return image_arr

def dense_optical_flow(frame, next):
    hsv = np.zeros((227,227,3))
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(frame,next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.array(hsv, dtype=np.uint8),cv2.COLOR_HSV2BGR)
    bgr = bgr / 255.0
    return bgr

def get_data(PATH, number, is_test, start_pos):
    '''
    Getting data from PATH
    
    Output:
        train_seq - list of images' sequences
    '''
    c = 0
    if (not is_test):
        seq = []
        ext = '/*.tif'
        # Lấy hết dữ liệu
        pos = 0
        for folder in os.listdir(PATH):
            if ('.' in folder):
                continue

            if (pos != start_pos):
                pos += 1
                continue
            
            img_seq = pims.ImageSequence(PATH + folder + ext)
            img_seq = normalize_image(list(img_seq))
            img_seq = np.asarray(img_seq)
            img_seq = np.expand_dims(img_seq, axis=3)
            seq.append(img_seq)
            c += 1
            if (c == number):
                break # Lấy một bộ dữ liệu
            pos += 1

        return np.asarray(seq)
    else:
        seq_test = []
        seq_gt = []
        ext_test = '/*.tif'
        ext_gt = '/*.bmp'
        
        pos = 0
        for folder in os.listdir(PATH):
            if ('.' in folder or '_gt' in folder):
                continue

            if (pos != start_pos):
                pos += 1
                continue
            
            img_seq = pims.ImageSequence(PATH + folder + ext_test)
            img_seq = normalize_image(list(img_seq))
            img_seq = np.asarray(img_seq)
            img_seq = np.expand_dims(img_seq, axis=3)
            seq_test.append(img_seq)
            
            if (folder + '_gt' in os.listdir(PATH)):
                print(folder + '_gt')
                img_seq_gt = pims.ImageSequence(PATH + folder + '_gt' + ext_gt)
                img_seq_gt = normalize_image(list(img_seq_gt))
                img_seq_gt = np.asarray(img_seq_gt)
                img_seq_gt = np.expand_dims(img_seq_gt, axis=3)
                seq_gt.append(img_seq_gt)
            else:
                img_seq_gt = np.full(img_seq.shape, 0)
                seq_gt.append(img_seq_gt)
                        
            c += 1
            if (c == number):
                break # Lấy một bộ dữ liệu
            pos += 1

        return (np.asarray(seq_test), np.asarray(seq_gt))
        
    
def create_data_train(sequence):
    X_new = None
    X_new_opflow = None
    for i in range(len(sequence)):
        X = sequence[i]
        X_tmp = np.zeros(shape=((len(X)-img_len),img_len,img_row,img_col,channel))
        X_tmp_opflow = np.zeros(shape=((len(X)-img_len),img_len,img_row,img_col,3))

        #X_tmp = np.zeros(shape=((len(X)-img_len+1),img_len,img_row,img_col,channel))
        for p in range(0, len(X)-img_len):
            X_tmp[p] = X[p:p+img_len]
            for l in range(img_len):
                X_tmp_opflow[p,l] = dense_optical_flow(X[p+l], X[p+l+1])
            
        if (i == 0):
            X_new = X_tmp
            X_new_opflow = X_tmp_opflow
        else:
            X_new = np.concatenate((X_new, X_tmp))
            X_new_opflow = np.concatenate((X_new_opflow, X_tmp_opflow))
            
        
    X_new = X_new.transpose((0,2,3,1,4))
    X_new_opflow = X_new_opflow.transpose((0,2,3,1,4))
    
    #mean_pixel = np.mean(X_new)
    #sd_pixel = np.std(X_new)
    
    return (X_new, X_new_opflow)

def create_data_test(sequence, gt_sequence):
    
    X_new = None
    y_new = None
    X_new_opflow = None

    for i in range(len(sequence)):
        X = sequence[i]
        X_gt = gt_sequence[i]
        
        X_tmp = np.zeros(shape=(len(X)-img_len,img_len,img_row,img_col,channel))
        X_tmp_opflow = np.zeros(shape=((len(X)-img_len),img_len,img_row,img_col,3))
        y_tmp = np.full(shape=(len(X)-img_len),fill_value=1)

        for p in range(0, len(X)-img_len):
            X_tmp[p] = X[p:p+img_len]
            for l in range(img_len):
                if (np.count_nonzero(X_gt[p+l]) != 0):
                    y_tmp[p] = -1
                X_tmp_opflow[p,l] = dense_optical_flow(X[p+l], X[p+l+1])

        if (i == 0):
            X_new = X_tmp
            y_new = y_tmp
            X_new_opflow = X_tmp_opflow
        else:
            X_new = np.concatenate((X_new, X_tmp))
            y_new = np.concatenate((y_new, y_tmp))
            X_new_opflow = np.concatenate((X_new_opflow, X_tmp_opflow))
        
    X_new = X_new.transpose((0,2,3,1,4))
    X_new_opflow = X_new_opflow.transpose((0,2,3,1,4))
    return (X_new, y_new, X_new_opflow)

def get_data_train_and_val(path, n_sets_use=5):
    n_sets = count_set(path)
    n_sets_use = min(n_sets, n_sets_use)
    length = 0
    for i in range(n_sets_use):
        length += get_data(path, 1, False,i)[0].shape[0] - img_len
    print(length)
    X_train = np.zeros((length,img_row,img_col,img_len,channel))
    X_train_opflow = np.zeros((length,img_row,img_col,img_len,3))


    run = 0
    for i in range(n_sets_use):
        train_seq = get_data(path, 1, False,i)
        X_tmp, X_tmp_opflow = create_data_train(train_seq)
        X_train[run:run+X_tmp.shape[0]] = X_tmp
        X_train_opflow[run:run+X_tmp.shape[0]] = X_tmp_opflow
        train_seq = []
        run += X_tmp.shape[0]

    n_val = len(X_train) // 10
    idx = random.sample(range(len(X_train)), n_val)
    X_val = X_train[idx]
    X_val_opflow = X_train_opflow[idx]

    X_train = np.delete(X_train, idx, axis=0)
    X_train_opflow = np.delete(X_train_opflow, idx, axis=0)
    print(X_train.shape, X_val.shape, X_train_opflow.shape, X_val_opflow.shape)
    return (X_train, X_val, X_train_opflow, X_val_opflow)

def get_data_test(path, n_sets_use=1):
    n_sets = count_set(path)
    n_sets_use = min(n_sets, n_sets_use)
    length = 0
    for i in range(n_sets_use):
        length += get_data(path, 1, True,i)[0][0].shape[0] - img_len
    print(length)

    X_test = np.zeros((length,img_row,img_col,img_len,channel))
    X_test_opflow = np.zeros((length,img_row,img_col,img_len,3))
    y_test_gt = np.zeros((length))

    run = 0
    for i in range(n_sets_use):
        test_seq, test_seq_gt = get_data(path, 1, True,i)
        X_tmp, y_tmp, X_tmp_opflow = create_data_test(test_seq, test_seq_gt)
        X_test[run:run+X_tmp.shape[0]] = X_tmp
        X_test_opflow[run:run+X_tmp.shape[0]] = X_tmp
        y_test_gt[run:run+X_tmp.shape[0]] = y_tmp
        test_seq = []
        test_seq_gt = []
        run += X_tmp.shape[0]

    print(X_test.shape, y_test_gt.shape, X_test_opflow.shape)
    return (X_test, y_test_gt, X_test_opflow)
