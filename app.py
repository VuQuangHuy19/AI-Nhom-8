import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Age & Gender Prediction")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")  # Màu nền giao diện

        # Label để hiển thị ảnh
        self.image_label = Label(root, bg="#f0f0f0")
        self.image_label.pack(pady=20)

        # Nút chọn ảnh (Thiết kế đẹp hơn)
        self.select_button = Button(
            root,
            text="📁 Chọn ảnh",
            command=self.select_image,
            bg="#4CAF50",  # Màu nền xanh lá
            fg="white",  # Màu chữ trắng
            font=("Arial", 12, "bold"),
            padx=20, pady=10,  # Kích thước nút
            relief="flat",  # Loại bỏ hiệu ứng nổi (border)
            border=0
        )
        self.select_button.pack(pady=10)

        # Nút suy luận (Thiết kế đẹp hơn)
        self.predict_button = Button(
            root,
            text="🤖 Dự đoán",
            command=self.predict_image,
            bg="#2196F3",  # Màu nền xanh dương
            fg="white",  # Màu chữ trắng
            font=("Arial", 12, "bold"),
            padx=20, pady=10,  # Kích thước nút
            relief="flat",  # Loại bỏ hiệu ứng nổi (border)
            border=0
        )
        self.predict_button.pack(pady=10)

        # Hiển thị kết quả
        self.result_label = Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
        self.result_label.pack(pady=20)

        self.selected_image_path = None  # Đường dẫn ảnh được chọn

        # Tải mô hình khi khởi chạy ứng dụng
        self.model = self.load_model()

    def load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, 20)  # 20 lớp (10 nhóm tuổi * 2 giới tính)
        checkpoint_path = "checkpoint/best.pt"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.to('cuda')
        model.eval()
        return model

    def select_image(self):
        # Mở hộp thoại chọn tệp
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.selected_image_path = file_path

            # Hiển thị ảnh trên giao diện
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize ảnh
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.result_label.config(text="")  # Xóa kết quả trước đó

    def preprocess_image(self, image_path):
        # Tiền xử lý ảnh
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1)) / 255  # Chuyển sang dạng (C, H, W) và chuẩn hóa
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch size
        image = torch.from_numpy(image).to('cuda').float()
        return image

    def predict_image(self):
        if not self.selected_image_path:
            self.result_label.config(text="Please select an image first!")
            return

        # Tiền xử lý ảnh
        image = self.preprocess_image(self.selected_image_path)

        # Danh sách các lớp
        categories = ['16-20 Female', '16-20 Male', '21-25 Female', '21-25 Male',
                      '26-30 Female', '26-30 Male', '31-35 Female', '31-35 Male',
                      '36-40 Female', '36-40 Male', '41-45 Female', '41-45 Male',
                      '46-50 Female', '46-50 Male', '51-55 Female', '51-55 Male',
                      '56-60 Female', '56-60 Male']

        # Tiến hành suy luận
        soft_max = nn.Softmax(dim=1)
        with torch.no_grad():
            output = self.model(image)
            prob = soft_max(output)

        # Lấy chỉ số của lớp có xác suất cao nhất
        max_value, max_index = torch.max(prob, dim=1)

        # Hiển thị kết quả
        predicted_class = categories[max_index.item()]
        confidence = max_value.item()
        self.result_label.config(
            text=f"Dự đoán: {predicted_class}\nĐộ tin ậy: {100 * confidence:.2f}%"
        )

# Chạy ứng dụng
if __name__ == "__main__":
    root = Tk()
    app = InferenceApp(root)
    root.mainloop()
