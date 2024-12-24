import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype


class AgeGenderDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        """
        Args:
            root (str): Thư mục chứa dữ liệu
            train (bool): Nếu True thì dùng dữ liệu huấn luyện, nếu False thì dùng dữ liệu kiểm tra
            transforms (callable, optional): Các biến đổi cho dữ liệu
        """
        self.root = root
        self.train = train
        self.transforms = transforms
        self.data = []  # Danh sách các đường dẫn hình ảnh
        self.labels = []  # Danh sách nhãn của hình ảnh
        self.age_mapping = ['16-20', '21-25', '26-30', '31-35', '36-40',
                            '41-45', '46-50', '51-55', '56-60']
        self.gender_mapping = ['Female', 'Male']

        # Tích hợp logic load data vào đây
        phase = 'train' if train else 'test'
        data_dir = os.path.join(root, phase)

        # Duyệt qua các nhóm độ tuổi và giới tính
        for age_group in os.listdir(data_dir):
            age_group_path = os.path.join(data_dir, age_group)
            if os.path.isdir(age_group_path):
                for gender in os.listdir(age_group_path):
                    gender_path = os.path.join(age_group_path, gender)
                    if os.path.isdir(gender_path):
                        for img_name in os.listdir(gender_path):
                            img_path = os.path.join(gender_path, img_name)
                            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(img_path)
                                self.labels.append((age_group, gender))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        age_group, gender = self.labels[idx]

        # Gán nhãn dựa trên index của age_group và gender
        age_label = self.age_mapping.index(age_group)
        gender_label = self.gender_mapping.index(gender)
        label = age_label * 2 + gender_label

        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Không thể đọc ảnh: {img_path}")

        # Đổi kênh màu từ BGR sang RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Áp dụng các transform nếu có
        if self.transforms:
            image = self.transforms(image)

        return image, label


# Các transform cho training và validation
train_transform = Compose([
    ToImage(),  # Chuyển đổi ảnh NumPy thành kiểu Image
    Resize((224, 224)),  # Resize ảnh
    ToDtype(torch.float32, scale=True),  # Chuyển đổi dtype và chuẩn hóa (scale)
])

val_transform = Compose([
    ToImage(),
    Resize((224, 224)),
    ToDtype(torch.float32, scale=True),
])