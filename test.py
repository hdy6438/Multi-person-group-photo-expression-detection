from tensorflow.keras.models import load_model

from setting import train_path, test_path, model_path
from tool import get_train_generator, get_test_generator, evaluate

if __name__ == "__main__":
    # 加载模型
    model = load_model(model_path)

    # 加载训练数据集
    train_generator = get_train_generator(train_path=train_path)
    # 加载测试数据集
    test_generator = get_test_generator(test_path=test_path)

    # 评估测试模型
    evaluate(model=model, train_generator=train_generator, test_generator=test_generator, to_txt=True)
