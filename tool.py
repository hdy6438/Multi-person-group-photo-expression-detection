import base64
import os
import random
import shutil
import time

import cv2
import numpy as np

import setting
from setting import input_size, batch_size, threshold
import math


def get_train_generator(train_path):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rotation_range=20,  # 随机旋转度数
        rescale=1 / 255,  # 数据归一化
        horizontal_flip=True,  # 水平翻转
        fill_mode='nearest',  # 填充方式,

    )
    return train_datagen.flow_from_directory(
        train_path,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
    )


# 加载测试数据集
def get_test_generator(test_path):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator(
        rescale=1. / 255  # 归一化
    ).flow_from_directory(
        test_path,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
    )


# 绘画
def draw(history):
    from matplotlib import pyplot as plt
    # accuracy的历史
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model/accuracy.png')

    plt.cla()

    # loss的历史
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model/loss.png')


# 评估模型
def evaluate(model, train_generator, test_generator, to_txt):
    print('#' * 20)
    print('train_scores；')
    train_scores = model.evaluate(train_generator, verbose=1)
    print(train_scores)
    print("{}:{}".format(model.metrics_names[0], train_scores[0]))
    print("{}:{}%".format(model.metrics_names[1], train_scores[1] * 100))

    print('#' * 20)
    print('test_scores；')
    test_scores = model.evaluate(test_generator, verbose=1)
    print(test_scores)
    print("{}:{}".format(model.metrics_names[0], test_scores[0]))
    print("{}:{}%".format(model.metrics_names[1], test_scores[1] * 100))

    if to_txt:
        with open('model/evaluate.txt', mode='w', encoding='utf-8') as f:
            f.write('#' * 30)
            f.write('\n')
            f.write('train_scores:')
            f.write('\n')
            f.write("{}:{}".format(model.metrics_names[0], train_scores[0]))
            f.write('\n')
            f.write("{}:{}%".format(model.metrics_names[1], train_scores[1] * 100))
            f.write('\n')
            f.write('#' * 30)
            f.write('\n')
            f.write('test_scores:')
            f.write('\n')
            f.write("{}:{}".format(model.metrics_names[0], test_scores[0]))
            f.write('\n')
            f.write("{}:{}%".format(model.metrics_names[1], test_scores[1] * 100))
            f.close()


def get_training_monitor():
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
    # 训练监控器
    return [
        # 模型检查,只保留最好的
        ModelCheckpoint(
            filepath='model/my_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            period=1,

        ),
        # 当指标不再改善时减低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=8,
            verbose=1,
            mode='auto',
            min_delta=0,
            cooldown=0,
            min_lr=0,
        ),
        # 模型可视化
        TensorBoard(
            log_dir='model/logs',
            histogram_freq=1,
            write_images=True,
            write_graph=True,
            write_grads=True,
        )

    ]


# 头像分割
def cut(frame, detector):
    from PIL import Image
    # 用cv2读取图片
    cv2_img = frame
    # 将图片输入dlib,检测人脸
    res = detector(cv2_img)
    # 用PIL读取图片,用以截取头像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    for k, d in enumerate(res):
        # 输入坐标截取头像
        region = pil_img.crop((d.left(), d.top(), d.right(), d.bottom())).resize(size=(input_size, input_size))
        region = np.array(region)
        faces.append(region)
    return faces


def predict(img, model):
    import numpy as np
    # 归一化
    img_tensor = img / 255.0
    # 图片扩充维度
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # 将图片输入模型
    prediction = model.predict(img_tensor)
    return prediction[0][0]


def data_transform(x):
    return (np.exp(x / threshold) - 1) / (math.e - 1)


def is_ok(frame, detector, model, person_num):
    faces = cut(frame=frame, detector=detector)  # 分割头像
    if len(faces) != person_num:
        return False, 0
    res_list = [predict(img=face, model=model) for face in faces]

    ok = len(list(filter(lambda x: x > threshold, res_list))) == 0
    res_list = data_transform(np.array(res_list))
    tot = 1 - np.mean(res_list)
    return ok, tot


def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def generate_random_str(randomlength=16):
    """
  生成一个指定长度的随机字符串
  """
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return str(int(time.time())) + random_str


def empty_model():
    if os.path.isdir("model"):
        shutil.rmtree("model")
        os.makedirs("model/logs")
        os.makedirs("model/jsmodel")
