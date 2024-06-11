from lib import *
from ViT_processor import ViT_Processor

def process_image(img):
    processor = ViT_Processor(img)
    grid = processor.run()

    return grid



# Tạo giao diện Gradio
iface = gr.Interface(
    fn=process_image,                # Hàm xử lý ảnh
    inputs=gr.Image(type="pil"),     # Đầu vào là một ảnh (kiểu PIL)
    outputs=gr.Image(type="pil"),    # Đầu ra là một ảnh (kiểu PIL)
    title="Image Processing App",    # Tiêu đề của ứng dụng
    description="Upload an image and get a processed image in return.",  # Mô tả ứng dụng
)
# Chạy ứng dụng
iface.launch(share=True)