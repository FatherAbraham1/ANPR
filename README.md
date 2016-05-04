# ANPR
___
Automatical Number Plate Recognition. A Java Version of EasyPR。  
ANPR是我的一个毕业设计项目，本来准备写一个雏形把过程过一遍写一份论文就得了，就跟着Mastering Opencv With Practical Computer Vision Projects上的示例项目写一了一遍，写完发现那识别成功率惨不忍睹，而且因为官方OpenCV3的Java接口的局限性很多实现必须借助其它工具于是决定重写这个程序。后来就谷歌到了EasyPR这个项目，这个项目已经有了Java版本，但是我发现[EasyPR-Java](https://github.com/fan-wenjie/EasyPR-Java)相对EasyPR已经有点老旧了，于是就打算照着EasyPR用Java重写了一遍。  
在此感谢[liuruoze](https://github.com/liuruoze)和他的[EasyPR](https://github.com/liuruoze/EasyPR)项目，特别是他的说明文章。

# 依赖
___
- [Weka](http://www.cs.waikato.ac.nz/ml/weka/) 机器学习库
- [JavaCSV](http://www.cs.waikato.ac.nz/ml/weka/) 操作CSV文件
- [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 支持向量机
- [OpenCV3](http://opencv.org/)
- [ImShow](https://github.com/master-atul/ImShow-Java-OpenCV)

# 微小的工作
___
目前已经把主体部分写完了，可以用Sobel算子处理车牌，支持向量机判别车牌，从车牌上分割字符，并用神经网络判别字符。  
接下来的工作是加入车牌颜色搜索部分，车牌颜色判断部分。并且加上liuruoze想加入的特征提取算法，并给程序添加一个GUI界面方便操作。  
不过在这之前先得把论文写完:(

# EasyPR
___
EasyPR是一个中文的开源车牌识别系统，其目标是成为一个简单、高效、准确的车牌识别引擎。

相比于其他的车牌识别系统，EasyPR有如下特点：

* 它基于openCV这个开源库。这意味着你可以获取全部源代码，并且移植到opencv支持的所有平台。
* 它能够识别中文。例如车牌为苏EUK722的图片，它可以准确地输出std:string类型的"苏EUK722"的结果。
* 它的识别率较高。图片清晰情况下，车牌检测与字符识别可以达到80%以上的精度。

___
### 版权
ANPR的源代码与训练数据遵循MIT协议开源  
和EasyPR有关的部分依照遵循Apache v2.0协议开源。  
特别的图像资源全部遵循EasyPR的版权声明。  
请确保在使用前了解以上协议的内容。

### 目录结构

以下表格是本工程中所有目录的解释:

|目录 | 解释
|------|----------
| src  | 所有源文件
| data | 所有训练数据和模型
| pics | 图片

以下表格是src目录下一些核心文件的解释与关系:

|文件 | 解释
|------|----------
| FINALS | 部分final变量
| getTrainingData | 获取训练数据
| Main | 主程序
| predictWeka | 支持向量机和神经网络判别
| segChars | 字符分割
| segPlate | 车牌定位
| utils | 一些工具函数
