import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import warnings
import argparse

def get_args():
    parser = argparse.ArgumentParser("Test Arguments")
    parser.add_argument('--checkpoint_path', '-cpp', type=str, default='checkpoint/best.pt')
    parser.add_argument('--image_path', '-i', type=str, required=True, help='Path to the image for inference')
    args = parser.parse_args()
    return args

def inference(args):
    # Đọc ảnh
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Unable to load image at path: {args.image_path}")
        return

    # Tiền xử lý ảnh
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, (2, 0, 1)) / 255  # Chuyển sang dạng (C, H, W) và chuẩn hóa
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch size
    image = torch.from_numpy(image).to('cuda').float()

    # Khởi tạo mô hình ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 20)  # Đầu ra là 20 lớp (10 nhóm tuổi * 2 giới tính)
    
    # Tải checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to('cuda')

    # Tiến hành suy luận
    categories = ['16-20 Female', '16-20 Male', '21-25 Female', '21-25 Male',
                  '26-30 Female', '26-30 Male', '31-35 Female', '31-35 Male',
                  '36-40 Female', '36-40 Male', '41-45 Female', '41-45 Male',
                  '46-50 Female', '46-50 Male', '51-55 Female', '51-55 Male',
                  '56-60 Female', '56-60 Male']  # Danh sách các lớp

    model.eval()
    soft_max = nn.Softmax(dim=1)
    with torch.no_grad():
        # Dự đoán
        output = model(image)
        prob = soft_max(output)

    # Lấy chỉ số của lớp có xác suất cao nhất
    max_value, max_index = torch.max(prob, dim=1)

    # In ra kết quả dự đoán
    print(f"Predicted class: {categories[max_index.item()]} with probability: {max_value.item():.2f}")

    # Hiển thị ảnh kết quả
    image_display = image.squeeze(0).cpu().numpy()
    image_display = np.transpose(image_display, (1, 2, 0))
    image_display = (image_display * 255).astype(np.uint8)
    image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)

    # Hiển thị hình ảnh và kết quả
    cv2.imshow(f"Prediction: {categories[max_index.item()]} : {max_value.item():.2f}", image_display)
    cv2.waitKey(0)  # Đợi cho đến khi người dùng nhấn phím
    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị ảnh

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = get_args()
    inference(args)
