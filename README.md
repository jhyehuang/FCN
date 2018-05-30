# 1、项目地址:https://github.com/jhyehuang/w10-huangzhijie-107158044.git

# 2、在convert_fcn_dataset.py中代码补充完整,生成的tfrecord文件：** 

    ![Image text](https://github.com/jhyehuang/w10-huangzhijie-107158044/blob/master/%E7%94%9F%E6%88%90%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)
# 3、使用google 免费GPU进行的训练,日志输出和训练过程请查看以下文件：** 
    w10_huangzhijie_107158044.ipynb


# 4、./eval下面生成验证的图片，其中val_xx_prediction.jpg的图片为模型输出的预测结果，内容应可以对应相应的annotation和img** 

# 5、在train.py文件中FCN-8s的代码已经补充完整** 

# 6、心得体会
## (1)、FCN-8s的实现

    FCN-8s借助VGG的第三层和第四层的信息，来帮助确定更精确的“位置信息”。

    FCN-8s从pool3，pool4和pool5中提取出信息，然后通过插值法调整大小把算出来的密度图简单相加。

    FCN-16s:先在pool4层添加一个卷积核大小为1*1的卷积层，产生一个额外的输出，然后在pool5
    （也就是网络执行过后的fc7输出）进行上采样步伐为2，让这个特征图回到pool4层所代表的信息，然后再
    把这两个输出进行融合，最后在融合后的结果上进行上采样步伐为16，这样就可以回到原图像尺寸大小，在
    这个特征图上进行像素分类。
    
    FCN-8s:就是先在pool3进行跟上面一样，添加卷积层产生额外的输出，然后再把FCN-16s中融合的结果进行上
    采样步长为2的上采样，这样在再把两个输出进行融合，接着是进行上采样步长为8的上采样，最后再对这个
    预测进行分类。


## (2)、对FCN的理解

        首先从文章的标题就可以看出文章使用了全卷积网络，那意思是说相对之前的alexnet模型来说，把最后的几层
    全连接层改成卷积层，网络也就变成了全卷积网络，以segmentation 的 ground truth作为监督信息，训练
    一个端到端的网络，让网络做pixelwise的prediction，直接预测label map。

        FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。与经典的
    CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN 可以接
    受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样,它恢复到输入图像相同的尺寸，
    从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐
    像素分类。
