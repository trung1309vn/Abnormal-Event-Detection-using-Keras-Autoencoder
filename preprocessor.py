import cv2
import os
import pims
import random
import matplotlib.pyplot as plt
import numpy as np

img_row = 227
img_col = 227
channel = 1
img_len = 9

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
            
            print(folder)
            
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
    for i in range(len(sequence)):
        X = sequence[i]
        X_tmp = np.zeros(shape=((len(X)-img_len),img_len,img_row,img_col,channel))

        #X_tmp = np.zeros(shape=((len(X)-img_len+1),img_len,img_row,img_col,channel))
        for p in range(0, len(X)-img_len):
            X_tmp[p] = X[p:p+img_len]
            
        if (i == 0):
            X_new = X_tmp
        else:
            X_new = np.concatenate((X_new, X_tmp))
        
    X_new = X_new.transpose((0,2,3,1,4))
    return X_new

def create_data_test(sequence, gt_sequence):
    
    X_new = None
    y_new = None

    for i in range(len(sequence)):
        X = sequence[i]
        X_gt = gt_sequence[i]
        
        X_tmp = np.zeros(shape=(len(X)-img_len,img_len,img_row,img_col,channel))
        y_tmp = np.full(shape=(len(X)-img_len),fill_value=1)

        for p in range(0, len(X)-img_len):
            X_tmp[p] = X[p:p+img_len]
            for l in range(img_len):
                if (np.count_nonzero(X_gt[p+l]) != 0):
                    y_tmp[p] = -1

        if (i == 0):
            X_new = X_tmp
            y_new = y_tmp
        else:
            X_new = np.concatenate((X_new, X_tmp))
            y_new = np.concatenate((y_new, y_tmp))
        
    X_new = X_new.transpose((0,2,3,1,4))
    return (X_new, y_new)

def get_data_train_and_val(path, n_sets_use=5):
    n_sets = count_set(path)
    n_sets_use = min(n_sets, n_sets_use)
    length = 0
    for i in range(n_sets_use):
        length += get_data(path, 1, False,i)[0].shape[0] - img_len
    print(length)
    X_train = np.zeros((length,img_row,img_col,img_len,channel))

    run = 0
    for i in range(n_sets_use):
        train_seq = get_data(path, 1, False,i)
        X_tmp = create_data_train(train_seq)
        X_train[run:run+X_tmp.shape[0]] = X_tmp
        train_seq = []
        run += X_tmp.shape[0]

    n_val = len(X_train) // 10
    idx = random.sample(range(len(X_train)), n_val)
    X_val = X_train[idx]
    X_train = np.delete(X_train, idx, axis=0)
    print(X_train.shape, X_val.shape)
    return (X_train, X_val)
'''
X_test = np.zeros((length,img_row,img_col,img_len,channel))
y_test_gt = np.zeros((length))

run = 0
for i in range(1):
    test_seq, test_seq_gt = get_data(TEST_PATH,1, True,i)
    X_tmp, y_tmp = create_data_test(test_seq, test_seq_gt)
    X_test[run:run+X_tmp.shape[0]] = X_tmp
    y_test_gt[run:run+X_tmp.shape[0]] = y_tmp
    test_seq = []
    test_seq_gt = []
    run += X_tmp.shape[0]


print(X_test.shape, y_test_gt.shape)
'''
def get_data_test(path, n_sets_use=1):
    n_sets = count_set(path)
    n_sets_use = min(n_sets, n_sets_use)
    length = 0
    for i in range(n_sets_use):
        length += get_data(path, 1, True,i)[0][0].shape[0] - img_len
    print(length)

    X_test = np.zeros((length,img_row,img_col,img_len,channel))
    y_test_gt = np.zeros((length))

    run = 0
    for i in range(n_sets_use):
        test_seq, test_seq_gt = get_data(path, 1, True,i)
        X_tmp, y_tmp = create_data_test(test_seq, test_seq_gt)
        X_test[run:run+X_tmp.shape[0]] = X_tmp
        y_test_gt[run:run+X_tmp.shape[0]] = y_tmp
        test_seq = []
        test_seq_gt = []
        run += X_tmp.shape[0]

    print(X_test.shape, y_test_gt.shape)
    return (X_test, y_test_gt)
