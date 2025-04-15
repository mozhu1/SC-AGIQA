import csv
import os
import pandas as pd
import numpy as np
from scipy import io
import torch.utils.data as data
from PIL import Image
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
def get_initial_patches(img_shape, patch_size):
    h, w = img_shape[:2]
    ph, pw = patch_size
    
    patches = [
        (0, 0, pw, ph),  # 左上
        (w - pw, 0, w, ph),  # 右上
        (0, h - ph, pw, h),  # 左下
        (w - pw, h - ph, w, h),  # 右下
        ((w - pw) // 2, (h - ph) // 2, (w + pw) // 2, (h + ph) // 2)  # 中间
    ]
    return patches

# 生成额外的裁剪区域，确保覆盖整个图像
def get_additional_patches(img_shape, patch_size, initial_patches, patch_num):
    h, w = img_shape[:2]
    ph, pw = patch_size
    
    extra_patches = []
    needed_patches = patch_num - len(initial_patches)
    step_x = (w - pw) // max(1, int(np.sqrt(needed_patches)))
    step_y = (h - ph) // max(1, int(np.sqrt(needed_patches)))
    
    for y in range(0, h - ph + 1, step_y):
        for x in range(0, w - pw + 1, step_x):
            if len(extra_patches) >= needed_patches:
                break
            extra_patches.append((x, y, x + pw, y + ph))
    
    return extra_patches
class KONIQDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        super(KONIQDATASET, self).__init__()

        self.data_path = root
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "koniq10k_scores_and_distributions.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["image_name"])
                mos = np.array(float(row["MOS_zscore"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for _, item in enumerate(index):
            for _ in range(patch_num):
                sample.append(
                    (os.path.join(root, "1024x768", imgname[item]), mos_all[item])
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVECDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None,is_train=True):
        #self.tokenizer = BertTokenizer.from_pretrained('/home/Newdisk/lq/IQAdatasets/livec/ChallengeDB_release/bert')
        #self.model = BertModel.from_pretrained('/home/Newdisk/lq/IQAdatasets/livec/ChallengeDB_release/bert')

        imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        imgpath = imgpath["AllImages_release"]
        imgpath = imgpath[7:1169]
        mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        labels = mos["AllMOS_release"].astype(np.float32)
        labels = labels[0][7:1169]
        if is_train==True:
            patch_num=patch_num//2
        sample = []
        #遍历图片
        for i, item in enumerate(index):
            # tuple.每个里面两个元素，一个图片路径一个评分。就是对一个图片采样patch_num次，后面再随机裁剪
            for aug in range(patch_num):
                sample.append(
                    (os.path.join(root, "Images", imgpath[item][0][0]), labels[item])
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        #text=self._load_text(path)
        return sample, target#,text

    def __len__(self):
        length = len(self.samples)
        return length


class UWIQADATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):

        imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        imgpath = imgpath["AllImages_release"]
        imgpath = imgpath[0:890]
        mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        labels = mos["AllMOS_release"].astype(np.float32)
        labels = labels[0][0:890]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append(
                    (os.path.join(root, "Images", imgpath[item][0][0]), labels[item])
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVEDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None,is_train=True):

        refpath = os.path.join(root, "refimgs")
        refname = getFileName(refpath, ".bmp")

        jp2kroot = os.path.join(root, "jp2k")
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, "jpeg")
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, "wn")
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, "gblur")
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, "fastfading")
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = io.loadmat(os.path.join(root, "dmos_realigned.mat"))
        labels = dmos["dmos_new"].astype(np.float32)

        orgs = dmos["orgs"]
        refnames_all = io.loadmat(os.path.join(root, "refnames_all.mat"))
        refnames_all = refnames_all["refnames_all"]

        refname.sort()
        if is_train==True:
            patch_num=patch_num//5
        sample = []

        for i in range(0, len(index)):
            train_sel = refname[index[i]] == refnames_all
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = "%s%s%s" % ("img", str(index), ".bmp")
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

###训练测试一个patch_num的
# class TID2013Dataset(data.Dataset):
#     def __init__(self, root, index, patch_num, transform=None):
#         refpath = os.path.join(root, "reference_images")
#         refname = getTIDFileName(refpath, ".bmp.BMP")
#         txtpath = os.path.join(root, "mos_with_names.txt")
#         fh = open(txtpath, "r")
#         imgnames = []
#         target = []
#         refnames_all = []
#         for line in fh:
#             line = line.split("\n")
#             words = line[0].split()
#             imgnames.append((words[1]))
#             target.append(words[0])
#             ref_temp = words[1].split("_")
#             refnames_all.append(ref_temp[0][1:])
#         labels = np.array(target).astype(np.float32)
#         refnames_all = np.array(refnames_all)

#         refname.sort()
#         sample = []
#         for i, item in enumerate(index):
#             train_sel = refname[index[i]] == refnames_all
#             train_sel = np.where(train_sel == True)
#             train_sel = train_sel[0].tolist()
#             for j, item in enumerate(train_sel):
#                 for aug in range(patch_num):
#                     sample.append(
#                         (
#                             os.path.join(root, "distorted_images", imgnames[item]),
#                             labels[item],
#                         )
#                     )
#         self.samples = sample
#         self.transform = transform

#     def _load_image(self, path):
#         try:
#             im = Image.open(path).convert("RGB")
#         except:
#             print("ERROR IMG LOADED: ", path)
#             random_img = np.random.rand(224, 224, 3) * 255
#             im = Image.fromarray(np.uint8(random_img))
#         return im

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self._load_image(path)
#         sample = self.transform(sample)
#         return sample, target

#     def __len__(self):
#         length = len(self.samples)
#         return length


# def getTIDFileName(path, suffix):
#     filename = []
#     f_list = os.listdir(path)
#     for i in f_list:
#         if suffix.find(os.path.splitext(i)[1]) != -1:
#             filename.append(i[1:3])
#     return filename
####不一个patch_num的
class TID2013Dataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, is_train=True):
        """
        Args:
            root (str): 数据集根目录，包含 reference_images、distorted_images 和 mos_with_names.txt。
            index (list): 用于筛选样本的索引列表（对应参考图的索引）。
            patch_num (int): 每个样本重复采样的次数。
            transform (callable, optional): 图像预处理函数。
            is_train (bool, optional): 是否为训练集。如果是训练集，采样次数为 patch_num // 3；否则为 patch_num。
        """
        # 参考图像路径
        refpath = os.path.join(root, "reference_images")
        # 获取所有参考图像名称
        refname = getTIDFileName(refpath, ".bmp.BMP")

        # 读取 MOS 文件，提取失真图像名称、参考图像名称和 MOS 分数
        txtpath = os.path.join(root, "mos_with_names.txt")
        with open(txtpath, "r") as fh:
            imgnames = []
            target = []
            refnames_all = []
            for line in fh:
                line = line.split("\n")
                words = line[0].split()
                imgnames.append(words[1])
                target.append(words[0])
                ref_temp = words[1].split("_")
                refnames_all.append(ref_temp[0][1:])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        # 根据 is_train 决定采样次数
        if is_train:
            sample_times = patch_num // 5  # 训练集采样次数为 patch_num // 3
        else:
            sample_times = patch_num  # 测试集采样次数为 patch_num

        # 构造样本
        sample = []
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for _ in range(sample_times):  # 根据 sample_times 决定采样次数
                    sample.append(
                        (
                            os.path.join(root, "distorted_images", imgnames[item]),
                            labels[item],
                        )
                    )
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


class CSIQDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, is_train=True):
        """
        Args:
            root (str): 数据集根目录，包含 src_imgs、dst_imgs_all 和 csiq_label.txt。
            index (list): 用于筛选样本的索引列表（对应参考图的索引）。
            patch_num (int): 每个样本重复采样的次数。
            transform (callable, optional): 图像预处理函数。
            is_train (bool, optional): 是否为训练集。如果是训练集，采样次数为 patch_num // 3；否则为 patch_num。
        """
        # 参考图像路径
        refpath = os.path.join(root, "src_imgs")
        # 获取所有参考图像名称
        refname = getFileName(refpath, ".png")

        # 读取标签文件，提取失真图像名称、参考图像名称和 MOS 分数
        txtpath = os.path.join(root, "csiq_label.txt")
        with open(txtpath, "r") as fh:
            imgnames = []
            target = []
            refnames_all = []
            for line in fh:
                line = line.split("\n")
                words = line[0].split()
                imgnames.append(words[0])
                target.append(words[1])
                ref_temp = words[0].split(".")
                refnames_all.append(ref_temp[0] + "." + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        # 根据 is_train 决定采样次数
        if is_train:
            sample_times = patch_num // 5  # 训练集采样次数为 patch_num // 3
        else:
            sample_times = patch_num  # 测试集采样次数为 patch_num

        # 构造样本
        sample = []
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for _ in range(sample_times):  # 根据 sample_times 决定采样次数
                    sample.append(
                        (
                            os.path.join(root, "dst_imgs_all", imgnames[item]),
                            labels[item],
                        )
                    )
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

from torchvision.transforms.functional import resize
from torchvision.transforms import ToTensor, Normalize




class SPAQDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None,is_train=True):
        super(SPAQDATASET, self).__init__()

        self.data_path = root
        anno_folder = os.path.join(self.data_path, "Annotations")
        xlsx_file = os.path.join(anno_folder, "MOS and Image attribute scores.xlsx")
        read = pd.read_excel(xlsx_file)
        imgname = read["Image name"].values.tolist()
        mos_all = read["MOS"].values.tolist()
        
        # Ensure MOS values are float32
        for i in range(len(mos_all)):
            mos_all[i] = np.array(mos_all[i]).astype(np.float32)

        sample = []
        test_image_dir = os.path.join(self.data_path, "TestImage")  # Define TestImage path
        if is_train==True:
            patch_num=patch_num//5
        for item in index:
            for _ in range(patch_num):
                sample.append(
                    (
                        os.path.join(test_image_dir, imgname[item]),  # Updated image path
                        mos_all[item],
                    )
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
            # 直接 Resize 到固定尺寸 512x384
            im = im.resize((512, 384), Image.BILINEAR)  # 修改为固定尺寸
        except Exception as e:
            print(f"ERROR IMG LOADED: {path}, Error: {e}")
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
            im = im.resize((512, 384), Image.BILINEAR)  # 错误时也统一尺寸
        return im

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class FBLIVEFolder(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None,is_train=True):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "labels_image.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["name"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)
        if is_train==True:
            patch_num=patch_num//5
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append(
                    (os.path.join(root, "database", imgname[item]), mos_all[item])
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
  
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class AGIQA3K(data.Dataset):
    def __init__(self, root, index=None, patch_num=10, transform=None, is_train=True):
        """
        Args:
            root (str): 数据集根目录，该目录下需包含 data.csv 文件，里面至少包含 "name", "mos_quality", "prompt" 和 "answer" 四列。
            index (list, optional): 用于筛选样本的索引列表（对应 CSV 文件行号）。
            patch_num (int, optional): 每个样本重复采样的次数。
            transform (callable, optional): 图像预处理函数。
            is_train (bool, optional): 是否为训练集。如果是训练集，采样次数为 patch_num // 15；否则为 patch_num。
        """
        # 读取 CSV 文件，提取图片名称、标签、prompt 和 answer
        csv_file = os.path.join(root, "data.csv")
        df = pd.read_csv(csv_file)
        
        # 检查并补全缺失值
        missing_mos = df['mos_quality'].isnull().sum()
        missing_prompt = df['prompt'].isnull().sum()
        missing_answer = df['answer'].isnull().sum()

        if missing_mos > 0:
            print(f"Missing 'mos_quality' values: {missing_mos}")
            df['mos_quality'] = df['mos_quality'].fillna(pd.Series(np.random.uniform(0, 10, size=len(df))))

        if missing_prompt > 0:
            print(f"Missing 'prompt' values: {missing_prompt}")
            df['prompt'] = df['prompt'].fillna("")

        if missing_answer > 0:
            print(f"Missing 'answer' values: {missing_answer}")
            df['answer'] = df['answer'].fillna("")

        img_names = df["name"].tolist()
        labels = df["mos_quality"].tolist()
        prompts = df["prompt"].tolist()
        answers = df["answer"].tolist()

        # 将 prompt 和 answer 连接起来，并记录分割位置
        prompt_answers = []
        for prompt, answer in zip(prompts, answers):
            combined = f"{prompt}|||{answer}"  # 使用 "|||" 作为分隔符
            split_index = len(prompt)  # 记录分割位置
            prompt_answers.append((combined, split_index))  # 存储连接后的字符串和分割位置

        # 如果提供 index 则筛选，否则全部使用
        if index is not None:
            img_names = np.array(img_names)[index].tolist()
            labels = np.array(labels)[index].tolist()
            prompt_answers = np.array(prompt_answers)[index].tolist()

        # 根据 is_train 决定采样次数
        if is_train:
            sample_times = patch_num // 5  # 训练集采样次数为 patch_num // 15
        else:
            sample_times = patch_num  # 测试集采样次数为 patch_num

        # 构造样本，重复 sample_times 次
        self.samples = []
        for i, name in enumerate(img_names):
            img_path = os.path.join(root, name)
            for _ in range(sample_times):
                self.samples.append((img_path, labels[i], prompt_answers[i]))

        self.transform = transform
        self.to_tensor = ToTensor()  # 用于将 PIL 图像转换为 Tensor
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化操作

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("ERROR IMG LOADED: ", path, "Error:", str(e))
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, idx):
        path, target, prompt_answer = self.samples[idx]  # 获取路径、标签和 prompt_answer
        combined, split_index = prompt_answer  # 解包连接后的字符串和分割位置

        # 加载图像
        image = self._load_image(path)

        # 未经过 transform 的图像，直接调整大小为 (224, 224)
        image_resized = resize(image, (224, 224))  # 调整大小
        image_resized = self.to_tensor(image_resized)  # 转换为 Tensor
        image_resized = self.normalize(image_resized)  # 归一化

        # 经过 transform 的图像
        if self.transform is not None:
            image_transformed = self.transform(image)
        else:
            image_transformed = self.to_tensor(image)  # 如果没有 transform，直接转换为 Tensor
            image_transformed = self.normalize(image_transformed)  # 归一化

        # 将两个图像的通道堆叠在一起
        image_combined = torch.cat([image_transformed, image_resized], dim=0)  # 在通道维度上堆叠

        # 返回堆叠后的图像、标签、连接后的字符串和分割位置
        return image_combined, target, combined

    def __len__(self):
        return len(self.samples)
import os
import pandas as pd
import numpy as np
from scipy import io
from PIL import Image

class AIGCIQA2023KDataset(data.Dataset):
    def __init__(self, root, index=None, patch_num=1, transform=None, is_train=True):
        """
        Args:
            root (str): 数据集根目录，该目录下需包含 "merged_output_aigciqa2023.csv" 和 "DATA/MOS/mosz1.mat" 文件。
            index (list, optional): 用于筛选样本的索引列表。
            patch_num (int, optional): 每个样本重复采样的次数。
            transform (callable, optional): 图片预处理函数。
            is_train (bool, optional): 是否为训练集。
        """
        # 读取 CSV 文件
        csv_file = os.path.join(root, "merged_output_aigciqa2023.csv")
        df = pd.read_csv(csv_file)
        
        # 检查并补全缺失值
        missing_answer = df['answer'].isnull().sum()
        missing_prompt = df['prompt'].isnull().sum()

        if missing_answer > 0:
            print(f"Missing 'answer' values: {missing_answer}")
            df['answer'] = df['answer'].fillna("")

        if missing_prompt > 0:
            print(f"Missing 'prompt' values: {missing_prompt}")
            df['prompt'] = df['prompt'].fillna("")

        img_names = df["name"].tolist()
        answers = df["answer"].tolist()
        prompts = df["prompt"].tolist()

        # 读取 mat 文件，假设变量名为 "mosz1"，并展平为一维数组
        mat_file = os.path.join(root, "DATA", "MOS", "mosz1.mat")
        mat_data = io.loadmat(mat_file)
        mos = mat_data["MOSz"].flatten()

        # 将 prompt 和 answer 连接起来，并记录分割位置
        prompt_answers = []
        for prompt, answer in zip(prompts, answers):
            combined = f"{prompt}|||{answer}"  # 使用 "|||" 作为分隔符
            split_index = len(prompt)  # 记录分割位置
            prompt_answers.append((combined, split_index))  # 存储连接后的字符串和分割位置

        # 如果提供 index 则筛选，否则全部使用
        if index is not None:
            img_names = np.array(img_names)[index].tolist()
            prompt_answers = np.array(prompt_answers)[index].tolist()
            mos = mos[index]

        # 根据 is_train 决定采样次数
        if is_train:
            sample_times = patch_num // 5  # 训练集采样次数为 patch_num // 5
        else:
            sample_times = patch_num  # 测试集采样次数为 patch_num

        # 构造样本，重复 sample_times 次
        self.samples = []
        for i, name in enumerate(img_names):
            # 将路径中的反斜杠替换为正斜杠
            name = name.replace("\\", "/")
            # 拼接完整路径：root/Image/name
            img_path = os.path.join(root, "Image", name)
            for _ in range(sample_times):
                self.samples.append((img_path, mos[i], prompt_answers[i]))

        self.transform = transform
        self.to_tensor = ToTensor()  # 用于将 PIL 图像转换为 Tensor
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化操作

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("ERROR IMG LOADED: ", path, "Error:", str(e))
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, idx):
        path, target, prompt_answer = self.samples[idx]  # 获取路径、标签和 prompt_answer
        combined, split_index = prompt_answer  # 解包连接后的字符串和分割位置

        # 加载图像
        image = self._load_image(path)

        # 未经过 transform 的图像，直接调整大小为 (224, 224)

        image_resized = resize(image, (224, 224))  # 调整大小
        image_resized = self.to_tensor(image_resized)  # 转换为 Tensor
        image_resized = self.normalize(image_resized)  # 归一化

        # 经过 transform 的图像
        if self.transform is not None:
            image_transformed = self.transform(image)
        else:
            image_transformed = self.to_tensor(image)  # 如果没有 transform，直接转换为 Tensor
            image_transformed = self.normalize(image_transformed)  # 归一化

        # 将两个图像的通道堆叠在一起
        image_combined = torch.cat([image_transformed, image_resized], dim=0)  # 在通道维度上堆叠

        # 返回堆叠后的图像、标签、连接后的字符串和分割位置
        return image_combined, target, combined

    def __len__(self):
        return len(self.samples)


    
    
    
class AGIQA1K(torch.utils.data.Dataset):
    def __init__(self, root, index=None, patch_num=10, transform=None, is_train=True):
        """
        Args:
            root (str): 数据集根目录，该目录下需包含 AIGC-1K_answer.csv 文件。
            index (list, optional): 用于筛选样本的索引列表（对应 CSV 文件行号）。
            patch_num (int, optional): 每个样本重复采样的次数。
            transform (callable, optional): 图像预处理函数。
            is_train (bool, optional): 是否为训练集。如果是训练集，采样次数为 patch_num // 15；否则为 patch_num。
        """
        # 读取 AIGC-1K_answer.csv 文件
        answer_csv_file = os.path.join(root, "AIGC-1K_answer.csv")
        df = pd.read_csv(answer_csv_file)

        # 检查并补全缺失值
        missing_mos = df['MOS'].isnull().sum()
        missing_prompt = df['prompt'].isnull().sum()
        missing_answer = df['answer'].isnull().sum()

        if missing_mos > 0:
            print(f"Missing 'MOS' values: {missing_mos}")
            df['MOS'] = df['MOS'].fillna(pd.Series(np.random.uniform(0, 10, size=len(df))))

        if missing_prompt > 0:
            print(f"Missing 'prompt' values: {missing_prompt}")
            df['prompt'] = df['prompt'].fillna("")

        if missing_answer > 0:
            print(f"Missing 'answer' values: {missing_answer}")
            df['answer'] = df['answer'].fillna("")

        # 提取数据
        img_names = df["name"].tolist()
        labels = df["MOS"].tolist()
        prompts = df["prompt"].tolist()
        answers = df["answer"].tolist()

        # 将 prompt 和 answer 连接起来，并记录分割位置
        prompt_answers = []
        for prompt, answer in zip(prompts, answers):
            combined = f"{prompt}|||{answer}"  # 使用 "|||" 作为分隔符
            split_index = len(prompt)  # 记录分割位置
            prompt_answers.append((combined, split_index))  # 存储连接后的字符串和分割位置

        # 如果提供 index 则筛选，否则全部使用
        if index is not None:
            img_names = np.array(img_names)[index].tolist()
            labels = np.array(labels)[index].tolist()
            prompt_answers = np.array(prompt_answers)[index].tolist()

        # 根据 is_train 决定采样次数
        if is_train:
            sample_times = patch_num // 5  # 训练集采样次数为 patch_num // 15
        else:
            sample_times = patch_num  # 测试集采样次数为 patch_num

        # 构造样本，重复 sample_times 次
        self.samples = []
        for i, name in enumerate(img_names):
            img_path = os.path.join(root, "file", name)  # 图片路径
            for _ in range(sample_times):
                self.samples.append((img_path, labels[i], prompt_answers[i]))

        self.transform = transform
        self.to_tensor = ToTensor()  # 用于将 PIL 图像转换为 Tensor
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化操作

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("ERROR IMG LOADED: ", path, "Error:", str(e))
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, idx):
        path, target, prompt_answer = self.samples[idx]  # 获取路径、标签和 prompt_answer
        combined, split_index = prompt_answer  # 解包连接后的字符串和分割位置

        # 加载图像
        image = self._load_image(path)

        # 未经过 transform 的图像，直接调整大小为 (224, 224)
        image_resized = resize(image, (224, 224))  # 调整大小
        image_resized = self.to_tensor(image_resized)  # 转换为 Tensor
        image_resized = self.normalize(image_resized)  # 归一化

        # 经过 transform 的图像
        if self.transform is not None:
            image_transformed = self.transform(image)
        else:
            image_transformed = self.to_tensor(image)  # 如果没有 transform，直接转换为 Tensor
            image_transformed = self.normalize(image_transformed)  # 归一化

        # 将两个图像的通道堆叠在一起
        image_combined = torch.cat([image_transformed, image_resized], dim=0)  # 在通道维度上堆叠

        # 返回堆叠后的图像、标签、连接后的字符串和分割位置
        return image_combined, target, combined

    def __len__(self):
        return len(self.samples)


class BIDDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        """
        Args:
            root (str): 数据集根目录，目录下需包含 ImageDatabase 文件夹和 DatabaseGrades.xls 文件
            index (list): 用于选择样本的索引列表
            patch_num (int): 每个样本采样的 patch 数量
            transform (callable, optional): 图像预处理函数
        """
        grades_file = os.path.join(root, "DatabaseGrades.xls")
        df = pd.read_excel(grades_file)
        # 假设 Excel 中有 "Image Number" 和 "Average Subjective Grade" 两列
        image_numbers = df["Image Number"].tolist()
        labels = df["Average Subjective Grade"].tolist()
        #print(image_numbers)
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                #print(item)
                # 构造图片文件名，假设图片命名为 "DatabaseImageXXXX.JPG"，其中 XXXX 为 4 位数字
                image_id = image_numbers[item]
                image_filename = f"DatabaseImage{int(image_id):04d}.JPG"
                image_path = os.path.join(root, "ImageDatabase", image_filename)
                sample.append((image_path, labels[item]))
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("ERROR IMG LOADED: ", path, "Error:", str(e))
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self._load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)