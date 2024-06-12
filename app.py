from lib import *
from ViT_Processor import ViT_Processor
from CLIP_Processor import CLIP_Processor

def process(image):

    process = ViT_Processor(image)
    process.run()

    path_results = "results images query.png"
    image_results = Image.open(path_results)

    if os.path.exists(path_results):
        os.remove(path_results)
        
    return image_results
    

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=process,
    inputs=gr.Image(type="pil"),  # Nhập ảnh dưới dạng đối tượng PIL
    outputs=[
        gr.Image(type="pil", label="Results of queried images"),
    ],
    title="Image Search Engine",
    description="Upload an image and get query images"
)
# Chạy ứng dụng
iface.launch(share=True)