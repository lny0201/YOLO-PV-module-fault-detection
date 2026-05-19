import shutil
import random
import os

# 原始路径
image_original_path = "D:\data\images"
label_original_path = "D:\data\labels"

cur_path = os.getcwd()
# 训练集路径
train_image_path = os.path.join(cur_path, "D:\datasets\images\\train")
train_label_path = os.path.join(cur_path, "D:\datasets\labels\\train")

# 验证集路径
val_image_path = os.path.join(cur_path, "D:\datasets\images\\val")
val_label_path = os.path.join(cur_path, "D:\datasets\labels\\val")

# 训练集和验证集列表文件路径
list_train = os.path.join(cur_path, "D:\datasets\\train.txt")
list_val = os.path.join(cur_path, "D:\datasets\\val.txt")

train_percent = 0.8
val_percent = 0.2


def del_file(path):
    for i in os.listdir(path):
        file_data = os.path.join(path, i)  # 使用os.path.join来避免操作系统差异
        os.remove(file_data)


def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    else:
        del_file(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    else:
        del_file(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    else:
        del_file(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
    else:
        del_file(val_label_path)


def clearfile():
    if os.path.exists(list_train):
        os.remove(list_train)
    if os.path.exists(list_val):
        os.remove(list_val)


def main():
    mkdir()
    clearfile()

    file_train = open(list_train, 'w')
    file_val = open(list_val, 'w')

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = list(range(num_txt))  # 使用list确保可以索引

    num_train = int(num_txt * train_percent)
    num_val = num_txt - num_train

    train = random.sample(list_all_txt, num_train)
    val = [i for i in list_all_txt if not i in train]

    print("训练集数目：{}, 验证集数目：{}".format(len(train), len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]  # 假设文件名以.txt结尾，且名称与图片对应（不包括扩展名）

        srcImage = os.path.join(image_original_path, name + '.jpg')
        srcLabel = os.path.join(label_original_path, name + ".txt")

        if i in train:
            dst_train_Image = os.path.join(train_image_path, name + '.jpg')
            dst_train_Label = os.path.join(train_label_path, name + '.txt')
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            file_train.write(dst_train_Image + '\n')
        else:
            dst_val_Image = os.path.join(val_image_path, name + '.jpg')
            dst_val_Label = os.path.join(val_label_path, name + '.txt')
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
            file_val.write(dst_val_Image + '\n')

    file_train.close()
    file_val.close()


if __name__ == "__main__":
    main()