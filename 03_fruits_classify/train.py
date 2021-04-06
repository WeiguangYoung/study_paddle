# 利用CNN实现水果分类
# 数据集：来自爬虫1036张水果图像
# 5个类别(苹果，香蕉，葡萄，橙子，梨)

import paddle
import paddle.fluid as fluid
import numpy
import os
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

name_dict = {"apple": 0, "banana": 1, "grape": 2,
             "orange": 3, "pear": 4}
data_root_path = "../dataset/fruits/"  # 数据集所在目录
test_file_path = data_root_path + "test.txt"  # 测试集
train_file_path = data_root_path + "train.txt"  # 训练集
# 记录每个类别有哪些图像 key:水果名称  value:所有图片路径列表
name_data_list = {}


# 将图片路径存入name_data_list字典
def save_train_test_file(path,  # 图片路径
                         name):  # 类别名称
    if name not in name_data_list:  # 类别不在字典中
        # 创建空列表，将图像放入列表，将列表放入字典
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list
    else:  # 类别在字典中，直接将图像路径加入列表
        name_data_list[name].append(path)


# 遍历数据集下子目录，将图片路径写入上面的字典
dirs = os.listdir(data_root_path)
for d in dirs:  # 遍历每个子目录
    full_path = data_root_path + d  # 拼接完整路径

    if os.path.isdir(full_path):  # 是目录
        imgs = os.listdir(full_path)
        for img in imgs:
            tmp_path = full_path + "/" + img
            save_train_test_file(tmp_path,
                                 d)  # 存入字典
    else:  # 是文件，跳过
        pass

# 划分测试集、训练集
with open(test_file_path, "w") as f:
    pass
with open(train_file_path, "w") as f:
    pass

# 遍历name_data_list字典，将内容写入测试集、训练集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 获取每个类别样本数量
    print("%s: %d张" % (name, num))

    for img in img_list:  # 从列表中取出每个图像路径
        if i % 10 == 0:  # 写测试集
            with open(test_file_path, "a") as f:
                # 拼接一行
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        else:
            with open(train_file_path, "a") as f:
                # 拼接一行
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        i += 1
print("数据预处理完成.")


def train_mapper(sample):
    """
    根据传入的样本数据(文本)读取图像内容并返回
    :param sample: 元组，格式为(图片路径, 类别)
    :return: 返回经过归一化的图像数据、类别
    """
    img, label = sample  # img为路径, lable为类别
    if not os.path.exists(img):
        print("图像不存在.")

    # 读取图片内容
    img = paddle.dataset.image.load_image(img)
    # 对图像进行变换，缩放到统一的大小
    img = paddle.dataset.image.simple_transform(
        im=img,  # 原始图像
        resize_size=100,  # 缩放大小100*100
        crop_size=100,  # 裁剪大小
        is_color=True,  # 是否为彩色图像
        is_train=True)  # 训练模式，随机裁剪
    # 对图像归一化处理，将每个像素转换到0~1之间
    # 归一化的作用可以提高模型稳定性、加快收敛速度
    img = img.astype("float32") / 255.0
    return img, label  # 返回图像数据、类别


# 原始读取器
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]

            for line in lines:  # 遍历每行
                # 去除换行符
                line = line.replace("\n", "")
                # 根据tab字符进行拆分
                img_path, lab = line.split("\t")

                yield img_path, int(lab)

    return paddle.reader.xmap_readers(
        train_mapper,  # reader读取的数据在这里二次处理
        reader,  # 读取数据函数
        cpu_count(),  # 线程数量，和逻辑CPU数量一致
        buffered_size)  # 缓冲区大小


# 定义reader
BATCH_SIZE = 32  # 批次大小
train_reader = train_r(train_list=train_file_path)
random_train_reader = paddle.reader.shuffle(
    reader=train_reader,
    buf_size=1300)  # 随机读取器
batch_train_reader = paddle.batch(
    random_train_reader,
    batch_size=BATCH_SIZE)  # 批量读取器
# 变量
image = fluid.layers.data(name="image",
                          shape=[3, 100, 100],
                          dtype="float32")
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")


# 组建CNN
# 结构： input --> 卷积/激活/池化 --> 卷积/激活/池化
#       --> 卷积/激活/池化 --> fc --> drop --> fc(softmax)
def convolution_neural_network(image, type_size):  # 类别数量
    # 第一组卷积/激活/池化
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,  # 原始图像
        filter_size=3,  # 卷积核大小3*3
        num_filters=32,  # 卷积核数量32个
        pool_size=2,  # 池化区域大小2*2
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1,
                                dropout_prob=0.5)
    # 第二组卷积/激活/池化
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 原始图像
        filter_size=3,  # 卷积核大小3*3
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小2*2
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2,
                                dropout_prob=0.5)

    # 第三组卷积/激活/池化
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 原始图像
        filter_size=3,  # 卷积核大小3*3
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小2*2
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3,
                                dropout_prob=0.5)
    # 全连接层
    fc = fluid.layers.fc(input=drop,
                         size=512,  # 输出值的个数
                         act="relu")
    # dropout
    drop = fluid.layers.dropout(x=fc,
                                dropout_prob=0.5)
    # 输出层(fc)
    predict = fluid.layers.fc(
        input=drop,
        size=type_size,  # 输出值个数，等于分类数量
        act="softmax")  # 输出层采用softmax激活函数
    return predict


# 创建VGG模型
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter,
                   groups, dropouts):
        # 创建conv2d,batch normal,dropout,pool2d组
        # 根据传入的组数量，反复执行卷积、BN,最后池化
        return fluid.nets.img_conv_group(
            input=ipt,  # 输入图像
            pool_stride=2,  # 池化步长值
            pool_size=2,  # 池化区域大小
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,  # 卷积核大小
            conv_act="relu",  # 激活函数
            conv_with_batchnorm=True,  # 是否使用批量归一化
            pool_type="max")  # 池化类型

    # 连续五组卷积池化
    conv1 = conv_block(image, 64, 2, [0.0, 0.0])
    conv2 = conv_block(conv1, 128, 2, [0.0, 0.0])
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0.0])
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0.0])
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0.0])

    drop = fluid.layers.dropout(x=conv5,
                                dropout_prob=0.2)
    fc1 = fluid.layers.fc(input=drop,
                          size=512,
                          act=None)
    bn = fluid.layers.batch_norm(input=fc1,
                                 act="relu")
    drop2 = fluid.layers.dropout(x=bn,
                                 dropout_prob=0.0)
    fc2 = fluid.layers.fc(input=drop2,
                          size=512,
                          act=None)
    predict = fluid.layers.fc(input=fc2,
                              size=type_size,
                              act="softmax")
    return predict


# 调用函数创建模型
# predict = convolution_neural_network(image=image,
#                                      type_size=5)

# 以下代码是使用VGG模型的代码
predict = vgg_bn_drop(image=image, type_size=5)

# 损失函数(交叉熵)
cost = fluid.layers.cross_entropy(
    input=predict,  # 预测值
    label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)
# 准确率
accuracy = fluid.layers.accuracy(
    input=predict,  # 预测值
    label=label)  # 真实值
# 优化器
# Adam优化器：兼顾稳定性和收敛速度
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 优化目标函数

# 执行器
place = fluid.CPUPlace()  # 运行在第一个GUP上
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder
feeder = fluid.DataFeeder(feed_list=[image, label],
                          place=place)

model_save_dir = "../model/fruits/"  # 模型保存目录
costs = []  # 记录损失值
accs = []  # 记录准确率
times = 0
batchs = []  # 迭代次数

# 开始训练
for pass_id in range(2):
    train_cost = 0  # 临时变量，存放训练损失值

    for batch_id, data in enumerate(batch_train_reader()):
        times += 1
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),  # 喂入参数
            fetch_list=[avg_cost, accuracy])
        # 打印
        if batch_id % 20 == 0:
            print("pass_id:%d, batch:%d, cost:%f, acc:%f"
                  % (pass_id, batch_id,
                     train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batchs.append(times)  # 记录迭代次数
# 训练结束后，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(
    dirname=model_save_dir,  # 保存的目录
    feeded_var_names=["image"],  # 预测时喂入的参数名称
    target_vars=[predict],  # 预测结果从哪里获取
    executor=exe)  # 执行器

print("训练保存模型完成.")

# 训练过程可视化
plt.title("trainning", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost/acc", fontsize=14)
plt.plot(batchs, costs, color="red", label="cost")
plt.plot(batchs, accs, color="blue", label="Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()

from PIL import Image


# 加载图像
def load_img(path):
    img = paddle.dataset.image.load_and_transform(
        path, 100, 100, False).astype("float32")
    img = img / 255.0  # 归一化
    return img


# 执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = "../model/fruits/"

infer_imgs = []  # 待预测图像数据
test_img = "grape_1.png"  # 待预测图像名称
# 读取图像，添加到待预测列表
infer_imgs.append(load_img(test_img))
infer_imgs = numpy.array(infer_imgs)  # 转换成数组

# 加载模型
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir,
                                  infer_exe)
# 执行预测
x_name = feed_target_names[0]  # 喂入参数名称
results = infer_exe.run(program=infer_program,
                        feed={x_name: infer_imgs},
                        fetch_list=fetch_targets)
print(results)
# 取第一张图像预测结果，并返回概率最大的值的索引
reulst = numpy.argmax(results[0])
for k, v in name_dict.items():
    if reulst == v:  # 如果预测结果等于数值
        print("预测结果:%s" % k)  # 打印名称

# 显示待预测图像
img = Image.open(test_img)
plt.imshow(img)
plt.show()
