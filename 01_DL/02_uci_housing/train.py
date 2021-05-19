# 波士顿房价预测案例
"""
1. 回归问题，使用一个fc网络(神经网络)作为模型执行预测
2. 输出值的个数为1，fc不采用任何激活函数
3. 损失函数：均方差
"""
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import matplotlib.pyplot as plt

# 第一步：数据准备、定义读取器
BUF_SIZE = 500  # 缓冲区大小
BATCH_SIZE = 20  # 批次大小

## reader
random_reader = paddle.reader.shuffle(
    paddle.dataset.uci_housing.train(),  # 训练集读取
    buf_size=BUF_SIZE)  # 随机读取器
train_reader = paddle.batch(random_reader,
                            batch_size=BATCH_SIZE)  # 批量读取器
## 打印样本数据
# for sample in train_reader():
#     print(sample)

x = fluid.layers.data(name="x", shape=[13], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")

# 第二步：创建网络模型、定义损失函数和优化器
y_pred = fluid.layers.fc(input=x,  # 输入，13个特征值
                         size=1,  # 输出值个数，预测1个价格
                         act=None)  # 回归问题不采用激活函数
## 损失函数
cost = fluid.layers.square_error_cost(input=y_pred,
                                      label=y)
avg_cost = fluid.layers.mean(cost)  # 均方差
## 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)  # 指定优化目标函数
## 克隆一个program用于模型评估(放在创建执行器之前)
# test_program = \
#     fluid.default_main_program().clone(for_test=True)

# 第三步：模型训练、评估
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
## feeder(数据喂入器)
feeder = fluid.DataFeeder(place=place,
                          feed_list=[x, y])

iter = 0
iters = []  # 记录迭代次数
train_costs = []  # 记录损失值
model_save_dir = "../model/uci_housing"  # 模型保存路径

for pass_id in range(120):
    train_cost = 0
    i = 0
    for data in train_reader():  # 循环从读取器中读取数据
        i += 1

        train_cost = exe.run(feed=feeder.feed(data),
                             fetch_list=[avg_cost])
        if i % 20 == 0:
            print("pass_id:%d, cost:%f" %
                  (pass_id, train_cost[0][0]))
        iter = iter + BATCH_SIZE  # 计算训练次数
        iters.append(iter)  # 记录训练次数
        train_costs.append(train_cost[0][0])  # 记录损失值

# 训练结束，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 模型目录不存在，则创建
fluid.io.save_inference_model(
    model_save_dir,  # 保存路径
    ["x"],  # 模型使用需要喂入参数
    [y_pred],  # 模型预测结果从哪个操作获取
    exe)  # 模型位于哪个执行器

# 训练过程可视化
plt.figure("Training Cost")
plt.title("Training Cost", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.plot(iters, train_costs, color="red", label="Training Cost")
plt.grid()
plt.savefig("./02_uci_housing/train.jpg")

# 第四步：预测、预测结果可视化
infer_exe = fluid.Executor(place)  # 用于预测的执行器
infer_result = []  # 存储预测结果
ground_truths = []  # 存储真实值

## 加载模型
## infer_program: 用于预测的program
## feed_target_name: 预测时需要传如的参数
## fetch_targets：预测结果从哪里获取
infer_program, feed_target_name, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir,  # 路径
                                  infer_exe)  # 加载到哪个执行器
# 从测试集读取数据
infer_reader = paddle.batch(
    paddle.dataset.uci_housing.test(),
    batch_size=200)  # 批量读取器
test_data = next(infer_reader())  # next只取一个批次数据
# 利用列表推导式取出13个特征和1个标签值
test_x = \
    np.array([data[0] for data in test_data]).astype("float32")
test_y = \
    np.array([data[1] for data in test_data]).astype("float32")
# 取出预测时，需要传入模型的参数的名称
# 取出加载模型返回的参数列表第一个参数
x_name = feed_target_name[0]
params = {x_name: test_x}  # 执行预测参数字典

results = infer_exe.run(
    program=infer_program,  # 预测的program
    feed=params,  # 参数字典
    fetch_list=fetch_targets)  # 根据加载模型的返回值取出结果
# print(results)

# 取出预测值，存入列表，可视化
for idx, val in enumerate(results[0]):
    # print("%d: %f" % (idx, val))
    infer_result.append(val)

# 取出真实值，填入列表，可视化
for idx, val in enumerate(test_y):
    # print("%d: %f" % (idx, val))
    ground_truths.append(val)

# 可视化
plt.figure("scatter")
plt.title("Infer", fontsize=24)
plt.xlabel("Ground Truth", fontsize=14)
plt.ylabel("Infer Result", fontsize=14)
# 绘制y=x斜线
x = np.arange(1, 30)
y = x
plt.plot(x, y)  # 绘制斜线
plt.scatter(ground_truths, infer_result,
            color="green", label="Test")
plt.grid()
plt.legend()
plt.savefig("./02_uci_housing/pred.png")  # 保存图像
plt.show()
