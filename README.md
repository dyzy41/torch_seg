# isprs_seg
ISPRS Vaihingen and Potsdam Datasets Segmentation

数据链接:  
https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/  
https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/  

现存功能:  
大尺寸遥感图像数据切片生成，滑窗切片  
大尺寸遥感图像滑窗预测，生成大尺寸预测结果  
完整的科研训练流程，支持生成各种tensorboard曲线图。包括learning_rate, train_loss, val_loss, val_acc(切片), val_acc(大图), f1  
完整的测试推理过程，支持tta预测，滑窗预测。  
支持netron生成网络模型可视化图。  
支持timm调用优化器，调用学习率迭代策略。  
支持timm调用backbone模型，自由组合分割算法模块和骨架网络模块。  
支持timm离线调用预训练模型。  
支持timm调用进行多波段数据训练。  
支持调用参数yaml文件训练。


未来工作:  
目标基于isprs数据采用了基础的数据增强方法，因此没有添加更多的数据增强策略。后面可以基于albumentations添加更多策略。


