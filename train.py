from tool import draw, evaluate, get_training_monitor, get_train_generator, get_test_generator, empty_model
from net import vgg19
from setting import *

if __name__ == "__main__":
    empty_model()

    # 创建神经网络

    model = vgg19()

    # 加载训练数据集
    train_generator = get_train_generator(train_path=train_path)
    # 加载测试数据集
    test_generator = get_test_generator(test_path=test_path)

    # 开始训练
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_training_monitor()  # 添加监控器
    )

    # 为history绘制图表
    draw(history=history)

    # 评估模型
    evaluate(model=model, train_generator=train_generator, test_generator=test_generator, to_txt=True)

# tensorboard --logdir=model/logs  --port=5555
