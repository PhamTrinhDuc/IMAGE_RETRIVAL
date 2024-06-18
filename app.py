from lib import *
from ViT_Processor import ViT_Processor
from CLIP_Processor import CLIP_Processor
from Module_RAG import llm
from ultralytics import YOLO

# QA_chain = llm.create_Chain_QA()
weight_path =  "weight_yolo/best_30epoch.pt"
model = YOLO(weight_path)


def process_image_user(image):

    path_query = "query_image.png"
    if os.path.exists(path_query): 
        os.remove(path_query)
    image.save(path_query)

    ######################## OUPUT FOR ViT RETRIEVAL #######################
    process = CLIP_Processor(image)
    path_images, image_names, distance_euclide = process.run()

    images_after_query = [Image.open(path_img) for path_img in path_images]
    grid_results_image = Image.open("results images query.png")

    # name_product = path_images[0].split("/")[1]
    # rewrite_prompt = prompt_user + f" .Một số sản phẩm {name_product} của bên tôi."
    # response = QA_chain.invoke({"query": rewrite_prompt})
    
    ######################## OUPUT FOR YOLO ###############################
    # xóa ảnh cũ trước khi detec ảnh mới.
    folder_results = "runs"
    if os.path.exists(folder_results):
        shutil.rmtree(folder_results)

    classes = model.names
    results = model.predict(path_query, verbose=False, save = True)
    path_image = "runs/detect/predict/" + path_query
    result_by_yolo = Image.open(path_image)

    class_name = None 
    scores = results[0].boxes.conf.cpu().numpy()
    # đảm bảo nếu model detect ra nhiều box thì lấy trung bình các score
    if len(scores) > 1:
        scores = sum(scores) / len(scores)

    if len(results[0].boxes.cls.numpy()) < 1: # không detect được object
        class_name = f"Chúng tôi không có sản phẩm nào như vậy"
    elif scores < 0.8: # score < 0.8
        class_name = f"Chúng tôi không có sản phẩm nào như vậy || {classes[int(results[0].boxes.cls.numpy()[0])]}"
    else:
        class_name = classes[int(results[0].boxes.cls.numpy()[0])]

    
    return f"Sản phẩm: {class_name} || {scores}", \
        result_by_yolo, \
        f"Sản phẩm: {image_names} || {distance_euclide}", \
        grid_results_image
        

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=process_image_user,

    inputs=[# gr.Textbox(lines=2, label="User", placeholder="Nhập yêu cầu của bạn tại đây."),
            gr.Image(type="pil", label="Nhập ảnh của bạn tại đây")],
    
    outputs=[
        gr.Textbox(lines=2, label="By YOLO_v8"),
        gr.Image(type="pil", label="By YOLO_v8"),
        gr.Textbox(lines=2, label="By ViT Retrival"),
        gr.Image(type="pil", label="By ViT Retrival"),
    ],
    title="Image Search Engine",
    description="Upload an image and get query images",
)
# Chạy ứng dụng
iface.launch(share=True)