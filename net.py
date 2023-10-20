from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications.vgg19 import VGG19

from setting import *


def my_print(text):
    print(text)
    f = open('model/model', mode='a')
    f.write(text)
    f.write('\n')
    f.close()


def vgg19():
    vgg19_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )  # 224*224*3
    vgg19_model.summary(print_fn=my_print)  # 查看网络

    # 搭建全连接层c
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg19_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))  # 输出层
    top_model.summary(print_fn=my_print)  # 查看网络

    model = Sequential()
    model.add(vgg19_model)
    model.add(top_model)
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])  # 优化器，损失函数
    model.summary(print_fn=my_print)  # 查看网络
    return model
