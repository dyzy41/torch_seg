这是一个基于pytorch的分割框架,由于做语义分割项目，设置了默认使用GPU计算  


现存功能:  
tools/cut_data.py: 大尺寸遥感图像数据切片生成，滑窗切片  
test_inference.py: 大尺寸遥感图像滑窗预测，生成大尺寸预测结果  
tools/merge_data.py: 基于切片预测结果还原成大图
完整的科研训练流程，支持生成各种tensorboard曲线图。包括learning_rate, train_loss, val_loss, val_acc(切片), val_acc(大图), f1  
vis_curve.py: 支持没有tensorboard的情况生成训练曲线  
完整的测试推理过程，支持tta预测，滑窗预测。  
支持netron生成网络模型可视化图。  
支持timm调用优化器，调用学习率迭代策略。  
支持timm调用backbone模型，自由组合分割算法模块和骨架网络模块。  
支持timm离线调用预训练模型。  
支持timm调用进行多波段数据训练。  
支持调用参数yaml文件训练。  
支持shell调用config.py脚本训练， python train.py config.yaml  
支持非常方便的生成训练和预测结果的文件夹，包括保存参数文件，曲线图，验证集预测结果，验证集精度
支持自定义对标签进行类别id处理  
支持多分类或二分类分割的输出类别转换  
网络模型目前的固定写法是networks内部的写法，方便添加网络模型结构  
数据集的输入按照data文件夹下的写法，使用write_file_list.py编写输入图片和标签图片的对应关系  

网络结构目前包含了：
unet，resunet，deeplab系列，swinunet，denseaspp等等，代码网络结构很容易扩展，与另一个仓库timm_seg进行对接。  

将来可以添加的：  
基于albumentations的数据增强策略  


训练步骤：  
修改config.yaml文件的root_path，这个是项目的总路径，保存文件的路径
执行./data/write_file_list.py,生成数据路径  
python train.py训练  
python test_dataloader.py测试并计算测试精度指标  




