from lib import *
from ViT_Processor import ViT_Processor
from CLIP_Processor import CLIP_Processor
from Module_RAG import llm

QA_chain = llm.create_Chain_QA()


def process_image_user(prompt_user, image):

    process = ViT_Processor(image)
    path_images = process.run()

    images_after_query = [Image.open(path_img) for path_img in path_images]
    name_product = path_images[0].split("/")[1]

    rewrite_prompt = prompt_user + f" .Một số sản phẩm {name_product} của bên tôi."

    response = QA_chain.invoke({"query": rewrite_prompt})

    return response['result'], \
        images_after_query[0], images_after_query[1], images_after_query[2], images_after_query[3], images_after_query[4]

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=process_image_user,

    inputs=[gr.Textbox(lines=2, label="User", placeholder="Nhập yêu cầu của bạn tại đây."),
            gr.Image(type="pil", label="Nhập ảnh của bạn tại đây")],
    
    outputs=[
        gr.Textbox(lines=2, label="Bot", placeholder="Câu trả lời của chúng tôi."),
        gr.Image(type="pil", label="Results of queried images"),
        gr.Image(type="pil", label="Results of queried images"),
        gr.Image(type="pil", label="Results of queried images"),
        gr.Image(type="pil", label="Results of queried images"),
        gr.Image(type="pil", label="Results of queried images")
    ],
    title="Image Search Engine",
    description="Upload an image and get query images",
)
# Chạy ứng dụng
iface.launch(share=True)