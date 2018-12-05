from tensorflow.contrib.keras.api.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.utils import to_categorical

from cifar_data_load import get_train_data_by_label

# 构建LeNet模型
# 这里使用的LeNet模型与原始的LeNet模型有些不同，除了最后一个连接层dense3使用了softmax激活外，在其他的层均未使用任何激活函数
def lenet(input):
    # 卷积层conv1
    conv1 = Conv2D(6, 5, (1, 1), 'valid', use_bias=True)(input)
    # 最大池化层maxpool1
    maxpool1 = MaxPool2D((2, 2), (2, 2), 'valid')(conv1)
    # 卷积层conv2
    conv2 = Conv2D(6, 5, (1, 1), 'valid', use_bias=True)(maxpool1)
    # 最大池化层maxpool2
    maxpool2 = MaxPool2D((2, 2), (2, 2), 'valid')(conv2)
    # 卷积层conv3
    conv3 = Conv2D(16, 5, (1, 1), 'valid', use_bias=True)(maxpool2)
    # 展开
    flatten = Flatten()(conv3)
    # 全连接层dense1
    dense1 = Dense(120, )(flatten)
    # 全连接层dense2
    dense2 = Dense(84, )(dense1)
    # 全连接层dense3
    dense3 = Dense(10, activation='softmax')(dense2)
    return dense3


if __name__ == '__main__':
    # 输入
    myinput = Input([32, 32, 3])
    # 构建网络
    output = lenet(myinput)
    # 建立模型
    model = Model(myinput, output)

    # 定义优化器，这里选用Adam优化器，学习率设置为0.0003
    adam = Adam(lr=0.0003)
    # 编译模型
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

    # 准备数据
    # 获取输入的图像
    X = get_train_data_by_label('data')
    # 获取图像的label，这里使用to_categorical函数返回one-hot之后的label
    Y = to_categorical(get_train_data_by_label('labels'))

    # 开始训练模型，batch设置为200，一共50个epoch
    model.fit(X, Y, 200, 50, 1, callbacks=[TensorBoard('./LeNet/log', write_images=1, histogram_freq=1)],
              validation_split=0.2, shuffle=True)
    # 保存模型
    model.save("lenet-no-activation-model.h5")