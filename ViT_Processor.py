from lib import *
from model import get_model
from helper_function import Helper

helper = Helper()


class ViT_Processor:
    def __init__(self, image_query):
        self.dataset_dict, self.image_filenames = helper.dataset_dict, helper.image_filenames
        self.index = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index.ntotal
        self.image_query = image_query
        self.model, self.processor = get_model("ViT")

    def get_image_embedding(self, images): # => process image => model => vector embedding
        inputs = self.processor(
            images,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = self.model(
                **inputs,
                output_hidden_states=True
        )
        # print(output.hidden_states[0].shape) => (1, 197, 768)
        # print(output.hidden_states[1].shape) => ...

        output = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
        # print(output.shape) => (1, 768)
        return output


    def embedding_image_database(self): # => store embedding of image in faiss
        SAVE_INTERVAL = 100
        os.makedirs("vector_db", exist_ok=True)
        path_index_json = "vector_db/index_ViT.json"
        path_index_bin = "vector_db/index_ViT.bin"


        if not os.path.exists(path_index_bin):
            for i, file in tqdm.tqdm(enumerate(self.image_filenames)):
                image = Image.open(file).convert("RGB")
                embedding = self.get_image_embedding(image)
                faiss.normalize_L2(embedding)
                self.index.add(embedding)

                if i % SAVE_INTERVAL == 0:
                    faiss.write_index(self.index, path_index_bin)
                faiss.write_index(self.index, path_index_bin)
        else:
            self.index = faiss.read_index(path_index_bin)

        if not os.path.exists(path_index_json):
            with open(path_index_json, "w") as f:
                json.dump(self.image_filenames, f)


    def Query(self, image_query: str, top_k: int): # => implement query
        query_embedding = self.get_image_embedding(image_query)
        faiss.normalize_L2(query_embedding)
        
        distance_euclide, indices = self.index.search(query_embedding, top_k)
        # print("Distance:", distance_euclide) # đã ranking
        # print("Indices:", indices)

        path_images = [self.image_filenames[i] for i in indices[0]]
        image_names = self.dataset_dict[path_images[0]]


        arg_distance = sum(distance_euclide[0]) / len(distance_euclide[0])
        if arg_distance < 0.7:
            return path_images, f"Chúng tôi không có sản phẩm như vậy || {image_names}", arg_distance
        
        return path_images, image_names, arg_distance


    def run (self):

        # embedding image
        self.embedding_image_database()
 
        # Đo lượng RAM sử dụng trước khi inference
        ram_before_infer = psutil.virtual_memory().used / (1024 ** 2)  # MB
        # query image
        path_images, image_names, distance_euclide = self.Query(self.image_query, top_k=5) # truy van anh
        # Đo lượng RAM sử dụng sau khi inference
        ram_after_infer = psutil.virtual_memory().used / (1024 ** 2)  # MB

        # plot image after query
        helper.plot_results(path_images)

        # print(f"RAM Used by Model ViT to Inference: {ram_after_infer - ram_before_infer:.2f} MB")
        return path_images, image_names, distance_euclide

if __name__ == "__main__":
    dataset_dir = "test_query"
    test_query = [os.path.join(dataset_dir, path_query) for path_query in os.listdir(dataset_dir)]
    image_query = Image.open("test_query/2ed921949deaddcb969d301a3f40f993_png_jpg.rf.60e03df4d14d3b19194e7c49b33ed106.jpg")

    # Show image query
    # plt.imshow(image_query)
    # plt.axis('off')
    # plt.show()


    processor = ViT_Processor(image_query=image_query)
    path_images, image_names, distance_euclide = processor.run()
    
    print(distance_euclide)
    print(image_names)