# -*- encoding:utf-8 -*-
import numpy as np
import pickle # python 序列化和反序列化工具


def unpickle(file):
    with open(file, 'rb')as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_photo(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2016]
    g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]
    b = np.reshape(b, [32, 32, 1])
    # 进行RGBd的重新排布
    photo = np.concatenate([r, g, b], -1)
    return photo


def get_train_data_by_label(label):
    batch_label = []
    labels = []
    data = []
    filenames = []
    for i in range(1,2):
        batch_label.append(unpickle("H:/柳博的空间/data/cifar-10-batches-py/data_batch_" + str(i))[b'batch_label'])
        labels = unpickle("H:/柳博的空间/data/cifar-10-batches-py/data_batch_" + str(i))[b'labels']
        data = unpickle("H:/柳博的空间/data/cifar-10-batches-py/data_batch_" + str(i))[b'data']
        filenames += unpickle("H:/柳博的空间/data/cifar-10-batches-py/data_batch_" + str(i))[b'filenames']

    data = np.concatenate(data, 0)
    label = str.encode(label)
    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
        for i in range(len(data)):
            array[i] = get_photo(data[i])
        return array
    elif label == b'labels':
        return labels
    elif label == b'batch_label':
        return batch_label
    elif label == b'filenames':
        return filenames
    else:
        raise NameError


def get_test_data_by_label(label):
    batch_label = []
    filenames = []

    batch_label.append(unpickle("H:/柳博的空间/data/cifar-10-batches-py")[b'batch_label'])
    labels = unpickle("H:/柳博的空间/data/cifar-10-batches-py")[b'labels']
    data = unpickle("H:/柳博的空间/data/cifar-10-batches-py")[b'data']
    filenames += unpickle("H:/柳博的空间/data/cifar-10-batches-py")[b'filenames']

    label = str.encode(label)
    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
        for i in range(len(data)):
            array[i] = get_photo(data[i])
        return array
        pass
    elif label == b'labels':
        return labels
        pass
    elif label == b'batch_label':
        return batch_label
        pass
    elif label == b'filenames':
        return filenames
        pass
    else:
        raise NameError
