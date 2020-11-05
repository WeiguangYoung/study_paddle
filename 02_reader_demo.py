# 1. 定义读取器，从样本文件中读取数据
# 2. 将原始读取器包装成随机读取器
# 3. 将随机读取器包装成批量读取器
import paddle


# 原始读取器，利用闭包结构实现
# 使用闭包的原因：灵活创建内容函数，对不同格式样本进行统一读取
def reader_creator(file_path):
    def reader():
        with open(file_path, "r") as f:  # 打开文件
            lines = f.readlines()  # 读取所有行
            for line in lines:
                yield line

    return reader


reader = reader_creator("dataset/test.txt")  # 原始读取器
shuffle_reader = paddle.reader.shuffle(reader, 10)  # 随机读取器
batch_reader = paddle.batch(shuffle_reader, 3)  # 批量读取器，3为批次大小
# for data in reader(): # reader是生成器函数，可迭代对象
# for data in shuffle_reader(): #shuffle_reader可迭代对象
for data in batch_reader():
    print(data)
