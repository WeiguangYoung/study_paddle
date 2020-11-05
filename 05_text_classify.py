# 文章分类示例
'''
数据来源：从网站上爬取56821条中文新闻摘要
数据类容：包含10类(国际、文化、娱乐、体育、财经、汽车、教育、科技、房产、证券)
'''

import os
from multiprocessing import cpu_count
import numpy as np
# import shutil
import paddle
import paddle.fluid as fluid

data_root = "dataset/news_classify/"
data_file = "news_classify_data.txt"
test_file = "test_list.txt"
train_file = "train_list.txt"
dict_file = "dict_txt.txt"

data_file_path = data_root + data_file  # 样本文件完整路径
dict_file_path = data_root + dict_file  # 字典文件完整路径
train_file_path = data_root + train_file  # 训练集文件完整路径
test_file_path = data_root + test_file  # 测试集文件完整路径


# 生成数据字典：把每一个汉字编码成一个数字，并存入字典文件
def create_dict():
    dict_set = set()
    with open(data_file_path, "r", encoding="utf-8") as f:  # 读取数据文件
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        title = line.split("_!_")[-1].replace("\n", "")
        # print("title:", title)
        for w in title:
            dict_set.add(w)  # 将文字添加到集合中
    # 把文字转换编码为数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])  # 文字-数字 映射添加到dict_list
        i += 1
        # print(s, ":", i)

    # 添加未知字符
    dict_txt = dict(dict_list)  # 将dict_list转换为字典
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)  # 将未知字符编码添加待编码字典中
    # 将字典保存到文件
    with open(dict_file_path, "w", encoding="utf-8") as f:
        f.write(str(dict_txt))
    print("生成数据字典完成!")


# 将一行原始文字转换为编码
def line_encoding(line, dict_txt, label):
    new_line = ""
    for w in line:
        if w in dict_txt:
            code = str(dict_txt[w])  # 从字典获取该字符编码
        else:
            code = str(dict_txt["<unk>"])
        new_line = new_line + code + ","
    new_line = new_line[:-1]  # 去掉最后一个逗号
    new_line = new_line + "\t" + label + "\n"
    return new_line


# 数据标记
def create_data_list():
    with open(test_file_path, "w") as f:  # 清空训练数据文件
        pass

    with open(train_file_path, "w") as f:  # 清空测试数据文件
        pass

    with open(dict_file_path, "r", encoding="utf-8") as f_dict:
        dict_txt = eval(f_dict.readlines()[0])  # 由文件生成字典

    with open(data_file_path, "r", encoding="utf-8") as f_data:  # 读入原始数据
        lines = f_data.readlines()

    # 将文章中每个字转换为编码，存入新的文件
    i = 0
    for line in lines:
        words = line.replace("\n", "").split("_!_")
        lable = words[1]  # 分类
        title = words[3]  # 标题
        if i % 10 == 0:  # 每10笔写入测试文件
            with open(test_file_path, "a", encoding="utf-8") as f_test:
                new_line = line_encoding(title, dict_txt, lable)
                f_test.write(new_line)
        else:  # 其它写入训练文件
            with open(train_file_path, "a", encoding="utf-8") as f_train:
                new_line = line_encoding(title, dict_txt, lable)
                f_train.write(new_line)
        i += 1
    print("生成训练、测试数据文件结束!")


create_dict()
create_data_list()


# 获取字典长度
def get_dict_len(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        line = eval(f.readlines()[0])  # 读取文件

    return len(line.keys())


# 创建数据读取器train_reader和test_reader
# 将传入数据由字符串转换为数字
def data_mapper(sample):
    data, label = sample
    val = [int(w) for w in data.split(",")]

    return val, int(label)


# 创建读取器train_reader, 每次读取一行编码后的数据
# 并拆分为标题(编码后的)、标记
def train_reader(train_file_path):
    def reader():
        with open(train_file_path, "r") as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱数据

            for line in lines:
                data, label = line.split("\t")
                yield data, label

    # 多线程下，使用自定义映射器 reader 返回样本到输出队列
    # 将mapper生成的数据交给reader进行二次处理，并输出
    return paddle.reader.xmap_readers(data_mapper,  # reader数据函数
                                      reader,  # 产生数据的reader
                                      cpu_count(),  # 处理样本的线程数(和CPU核数一样)
                                      1024)  # 数据缓冲队列大小


# 创建数据读取器test_reader
# 和train_reader读取源文件不同，不需要随机打乱数据
def test_reader(test_file_path):
    def reader():
        with open(test_file_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                data, label = line.split("\t")
                yield data, label

    return paddle.reader.xmap_readers(data_mapper,  # reader数据函数
                                      reader,  # 产生数据的reader
                                      cpu_count(),  # 处理样本的线程数(和CPU核数一样)
                                      1024)  # 数据缓冲队列大小


# 定义网络
def CNN_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=98):
    # embeding(词向量)层：将高度稀疏的离散输入嵌入到一个新的实向量空间
    # 以使用更少的维度，表示更丰富的信息
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # 第一个卷积、池化层
    # sequence_conv_pool: 序列卷积、池化层构成
    conv_1 = fluid.nets.sequence_conv_pool(input=emb,  # 输入
                                           num_filters=hid_dim,  # 卷积核数目
                                           filter_size=3,  # 卷积核大小
                                           act="tanh",  # 激活函数
                                           pool_type="sqrt")  # 池化类型

    conv_2 = fluid.nets.sequence_conv_pool(input=emb,  # 输入
                                           num_filters=hid_dim2,  # 卷积核数目
                                           filter_size=4,  # 卷积核大小
                                           act="tanh",  # 激活函数
                                           pool_type="sqrt")  # 池化类型

    output = fluid.layers.fc(input=[conv_1, conv_2], size=class_dim, act="softmax")
    return output


# 定义和训练模型
EPOCH_NUM = 40
model_save_dir = "model/news_classify/"

# 定义输入数据
words = fluid.layers.data(name="words", shape=[1], dtype="int64",
                          lod_level=1)  # 张量层级
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# 获取字典长度
dict_dim = get_dict_len(dict_file_path)
# 调用函数，生成卷积神经网络
model = CNN_net(words, dict_dim)
# 定义损失函数
cost = fluid.layers.cross_entropy(input=model, label=label)  # 交叉熵作为损失函数
avg_cost = fluid.layers.mean(cost)
# 计算准确率
acc = fluid.layers.accuracy(input=model,  # 输入：即预测网络
                            label=label)  # 数据集标签

# 复制program用于测试
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化器
# Adaptive Gradient(自适应梯度下降优化), 对于不同参数自动调整梯度大小
# 对于数据较为稀疏的特征梯度变大，数据较为稠密的特征梯度减小
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)  # 求avg_cost最小值

# 创建执行器
# place = fluid.CPUPlace()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化系统参数

# 准备数据
## 训练数据读取器
tr_reader = train_reader(train_file_path)
train_reader = paddle.batch(reader=tr_reader, batch_size=128)
## 测试数据读取器
ts_reader = test_reader(test_file_path)
test_reader = paddle.batch(reader=ts_reader, batch_size=128)

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])  # feeder

# 开始训练
for pass_id in range(EPOCH_NUM):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每100次打印一次cost, acc
        if batch_id % 100 == 0:
            print("pass_id:%d, batch_id:%d, cost:%0.5f, acc:%0.5f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 使用测试数据集测试
    test_costs_list = []
    test_accs_list = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_costs_list.append(test_cost[0])
        test_accs_list.append(test_acc[0])

    # 计算平均损失值和准确率
    avg_test_cost = (sum(test_costs_list) / len(test_costs_list))
    avg_test_acc = (sum(test_accs_list) / len(test_accs_list))

    print("pass_id:%d, test_cost:%0.5f, test_acc:%0.5f" %
          (pass_id, avg_test_cost, avg_test_acc))

# 保存模型
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
else:
    fluid.io.save_inference_model(model_save_dir,  # 模型保存的目录
                                  feeded_var_names=[words.name],  # 需要喂入的数据
                                  target_vars=[model],  # 从哪里的到预测结果
                                  executor=exe)  # 执行器
print("模型保存完成!")


# 将句子转换为编码
def get_data(sentence):
    with open(dict_file_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0])

    # dict_txt = dict(dict_txt)
    keys = dict_txt.keys()
    ret = []
    for s in sentence:
        if not s in keys:
            s = "<unk>"
        ret.append(int(dict_txt[s]))
    return ret


# 读取模型

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

print("加载模型:", model_save_dir)
infer_program, feeded_var_names, target_var = \
    fluid.io.load_inference_model(dirname=model_save_dir, executor=exe)
# 生成测试数据
texts = []
data1 = get_data("在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说")
data2 = get_data("综合'今日美国'、《世界日报》等当地媒体报道，芝加哥河滨警察局表示")
data3 = get_data("中国队无缘2020年世界杯")
data4 = get_data("中国人民银行今日发布通知，提高准备金率，预计释放4000亿流动性")
data5 = get_data("10月20日,第六届世界互联网大会正式开幕")

texts.append(data1)
texts.append(data2)
texts.append(data3)
texts.append(data4)
texts.append(data5)

# 获取每个句子词数量
base_shape = [[len(c) for c in texts]]
# 生成预测数据
tensor_words = fluid.create_lod_tensor(texts, base_shape, place)
# 执行预测
# tvar = np.array([target_var], dtype="int64")
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)
# fetch_list=tvar)

names = ["文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "国际", "证券"]

# 获取结果概率最大的label
for i in range(len(texts)):
    lab = np.argsort(result)[0][i][-1]
    print("预测结果: %d, 名称:%s, 概率:%f" %
          (lab, names[lab], result[0][i][lab]))
