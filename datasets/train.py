
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/data4/zheli/yolov10-main/datasets/yolov10n.pt')
    model.train(data= r'/data4/zheli/yolov10-main/datasets/data.yaml',
                
                imgsz=640,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=64,
                close_mosaic=10,
                workers=0,
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )


# from ultralytics import YOLO
# from ultralytics import YOLO
# from sklearn.metrics import f1_score
# import torch
# import numpy as np
#
#
# def extract_labels(preds, conf_thres=0.25, iou_thres=0.45):
#     # preds 是一个包含模型输出的列表，每个元素是一个字典，包含边界框、置信度和类别等信息
#     labels = []
#     for pred in preds:
#         # 过滤掉置信度低于conf_thres的预测
#         pred = [p for p in pred if p['conf'] > conf_thres]
#         # 对每个预测，提取类别标签
#         for p in pred:
#             labels.append(int(p['cls'].cpu().numpy()))  # 假设'cls'是类别索引
#     return labels
#
#
# def custom_validation(model, dataloader, conf_thres=0.25, iou_thres=0.45):
#     model.eval()
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for imgs, targets, paths, shapes in dataloader:
#             imgs = imgs.to(model.device)
#             preds = model(imgs, augment=False)[0]
#             pred_labels = extract_labels(preds, conf_thres, iou_thres)
#             all_preds.extend(pred_labels)
#
#             # 假设targets已经是一个包含类别标签的列表
#             # 如果不是，您需要根据实际情况调整这部分代码
#             all_targets.extend(targets)
#             # 计算F1分数
#     if len(all_preds) > 0 and len(all_targets) > 0:
#         f1 = f1_score(all_targets, all_preds, average='weighted')
#         print(f'Validation F1 Score: {f1:.4f}')
#     else:
#         print('No predictions or targets to calculate F1 score.')
#
# if __name__ == '__main__':
#     model = YOLO('yolov10n.pt')
#     train_params = {
#         'data': r'D:\yolov10-main\datesets\red\data.yaml',  # 注意这里我修正了 'datesets' 为 'datasets'
#         'cache': False,
#         'imgsz': 640,
#         'epochs': 30,
#         'batch': 4,
#         'device': 'cpu',  # 或'0'（如果使用GPU）
#         'single_cls': False,  # 是否是单类别检测
#         'close_mosaic': 10,
#         'workers': 0,
#         'optimizer': 'SGD',  # 在这里添加了逗号
#         'amp': True,
#         'project': 'runs/train',
#         'name': 'exp',
#     }
#     model.train(**train_params)
#     # 验证模型（假设您已经有了一个适当的val_dataloader）
#     # 这里需要您自己定义和构建val_dataloader
#     # val_dataloader = ...
#     from ultralytics import YOLO
#     from torch.utils.data import DataLoader
#
#     # 假设您已经有一个YOLO模型实例
#     model = YOLO('yolov10n.pt')
#
#     # 加载验证数据集的配置文件
#     val_data_path = r'D:\yolov10-main\datesets\red\data.yaml'  # 替换为您的验证数据集配置文件路径
#
#     # 使用Ultralytics YOLO库的数据加载功能
#     # 注意：这里的代码可能需要根据您使用的YOLO版本和API进行调整
#     # 假设YOLO类有一个from_pretrained方法，并且该方法接受一个配置文件路径来加载数据集
#     # 如果不是，请查阅Ultralytics YOLO的文档来了解如何正确加载数据集
#     # 下面的代码是一个假设性的示例，您需要根据实际情况进行调整
#     val_dataset = YOLO.from_pretrained(val_data_path, train=False)  # 假设的API调用
#
#     # 由于Ultralytics YOLO的API可能会变化，以下是一个更通用的方法来加载数据集
#     # 您可能需要自己实现一个Dataset类，或者使用Ultralytics提供的Dataset类（如果有的话）
#     # 这里我们假设已经有一个名为ValDataset的类，它继承自torch.utils.data.Dataset
#     # 并且已经实现了__getitem__和__len__方法
#     # 由于我无法直接访问Ultralytics YOLO的内部实现，以下是一个简化的DataLoader创建过程
#     # 您需要根据实际情况替换ValDataset为正确的数据集类
#     # 并且设置正确的batch_size和其他参数
#     batch_size = 4  # 根据您的GPU内存和训练需求调整批量大小
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
#     # 现在您可以使用val_dataloader来验证模型了
#     custom_validation(model, val_dataloader)


