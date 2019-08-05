# cats_vs_dogs
深度学习-猫狗大战App实现
---

### 项目说明
---

本项目是优达学城的一个毕业项目。项目要求使用深度学习方法识别一张图片是猫还是狗

- 输入：一张彩色图片
- 输出：是猫还是狗


### 项目环境
---
项目使用Anaconda搭建环境。mac os、jupyter notebook、python3、keras、xcode等。单模型训练，个人mac就行；融合模型训练使用GPU云服务。

代码文件说明：

- detection_remove.ipynb -- 异常图片剔除
- cat_vs_dog.ipynb --- 融合模型实现（ResNet50, Xception, InceptionV3，InceptionResNetV2 四个模型）
- cat_vs_dog_cam.ipynb -- iOS实现猫狗识别App（ResNet50单模型）

### 数据来源
---
数据集来自 kaggle 上的一个竞赛：Dogs vs. Cats Redux: Kernels Edition。
下载kaggle猫狗数据集解压后分为 3 个文件 train.zip、 test.zip 和 sample_submission.csv。
train 训练集包含了 25000 张猫狗的图片， 每张图片包含图片本身和图片名。命名规则根据“type.num.jpg”方式命名。
test 测试集包含了 12500 张猫狗的图片， 每张图片命名规则根据“num.jpg”，需要注意的是测试集编号从 1 开始， 而训练集的编号从 0 开始。
sample_submission.csv 需要将最终测试集的测试结果写入.csv 文件中，上传至 kaggle 进行打分。

### 基准模型
---

1. 融合模型实现：
项目使用ResNet50, Xception, InceptionV3，InceptionResNetV2 四个模型完成。本项目的最低要求是 kaggle Public Leaderboard 前10%。在kaggle上，总共有1314只队伍参加了比赛，所以需要最终的结果排在131位之前，131位的得分是0.06127，所以目标是模型预测结果要小于0.06127。

2. ResNet50单模型，在iphone手机上运行猫狗识别模型

### 评估指标
---

kaggle 官方的评估标准是 LogLoss，下面的表达式就是二分类问题的 LogLoss 定义。

$$ LogLoss = -\frac{1}{n}\sum_{i=1}^n [y_ilog(\hat{y}_i)+(1-y_i)log(1- \hat{y}_i)]$$

其中：

- n 是测试集中图片数量
- $\hat{y}_i$ 是图片预测为狗的概率
- $y_i$ 如果图像是狗，则为1，如果是猫，则为0
- $log()$ 是自然（基数 $e$）对数

对数损失越小，代表模型的性能越好。上述评估指标可用于评估该项目的解决方案以及基准模型。

### 融合模型设计大纲
---
 - 本项目使用融合模型实现。
 - 融合模型方法：首先将特征提取出来，然后拼接在一起，构建一个全连接分类器训练即可。
 - 模型融合能提供成绩的理论依据是，有些模型识别狗的准确率高，有一些模型识别猫的准确率高，给这些模型不同的权重，让他们能够取长补短，强强联合，综合各自的优势，为了更高的融合模型，可以提取特征进行融合，这样会有更好的效果，弱特征的权重会越学越小，强特征会越学越大，最后得到效果非常好的模型。
 
 ![model_png](https://github.com/bjheweihua/cats_vs_dogs/blob/master/source/model.png "")

该模型使用云端GPU训练实现。

**1. 数据预处理**

- 从kaggle下载好图片
- 将猫和狗的图片解压分别放在不同的文件夹以示分类，使用创建符号链接的方法
- 对图片进行resize，保持输入图片信息大小一致
- 图像文件分类后的路径如下：
- image
- ├── test 
- ├── train 
- ├── img_train
- │   ├── cat 
- │   └── dog 

**2. 模型搭建**

Kera的应用模块Application提供了带有预训练权重的Keras模型，这些模型可以用来进行预测、特征提取和微调整和。

- ResNet50 默认输入图片大小是 `224*224*3`
- Xception 默认输入图片大小是 `299*299*3`
- InceptionV3 默认输入图片大小是 `299*299*3`
- InceptionResNetV2 默认输入图片大小是 `299*299*3`

在Keras中载入模型并进行全局平均池化，只需要在载入模型的时候，设置`include_top=False`, `pooling='avg'`. 每个模型都将图片处理成一个` 1*2048 `的行向量，将这四个行向量进行拼接，得到一个` 1*8192 `的行向量， 作为数据预处理的结果。


**3. 模型训练&模型调参**

载入预处理的数据之后，先进行一次概率为0.5的dropout, 减少参数减少计算量，防止过拟合，然后直接连接输出层，激活函数为Sigmoid，优化器为Adadelta，输出一个零维张量，表示某张图片中有狗的概率。

**4. 模型评估**

- 使用$Logloss$进行模型评估,上传Kaggle判断是否符合标准

**5. 可视化**

- 进行数据探索并且可视化原始数据
- 可视化模型训练过程的准确率曲线，损失函数曲线等

**6. 模型调优**

- 训练时，使用交叉验证，打印acc和loss；观察训练结果，使用测试集验证结果；要提高模型效果，可以对训练集进行数据增强，尽量剔除异常图片，排除一些图片对模型的学习掌握，使用更强大的算法来优化模型的表现，例如最近出来的EfficientNets网络等。 大胆假设，小心求证。


### iOS实现猫狗识别App
---
本项目同时也实现了使用 Keras 和 Xcode 搭建了一个基于iOS平台的深度学习-猫狗识别App，可以通过App摄像头或者相册输入一张彩色猫或者狗的图片预测是猫或者狗的概率。

具体使用请看cat_vs_dog_cam.ipynb

如果不想搭建环境复现实验结果，你可以下载VisionMLDemo，里面有已经训练好的模型以及代码可以运行.

猫狗预测效果如下：

![cat_vs_dog](https://github.com/bjheweihua/cats_vs_dogs/blob/master/source/cat_vs_dog_ios.gif "")
