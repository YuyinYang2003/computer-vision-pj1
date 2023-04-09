import numpy as np
import struct
'''导入MNIST数据集'''

train_image_file = './data/train-images.idx3-ubyte'
train_label_file = './data/train-labels.idx1-ubyte'
test_image_file = './data/t10k-images.idx3-ubyte'
test_label_file = './data/t10k-labels.idx1-ubyte'

def get_image(path):
    #打开train_image_file和test_image_file路径，将每张28*28的图片flatten成1*784向量
    #最后输出N*784矩阵，N为数据集图片数
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, 784)
        images = np.array(images, dtype = float)
    return images

def get_label(path):
    #打开train_label_file和test_label_file路径，读取每张图片的label
    #最后输出N*1矩阵，N为数据集图片数
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II',f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        labels = np.array(labels, dtype = float)
    return labels

def load_dataset():
    #输出(train_image, train_label, val_image, val_label, test_image, test_label)元组
    train_image = get_image(train_image_file)[:50000]
    train_label = get_label(train_label_file)[:50000]
    val_image = get_image(train_image_file)[50000:]
    val_label = get_label(train_label_file)[50000:]
    test_image = get_image(test_image_file)
    test_label = get_label(test_label_file)
    return (train_image, train_label, val_image, val_label, test_image, test_label)
