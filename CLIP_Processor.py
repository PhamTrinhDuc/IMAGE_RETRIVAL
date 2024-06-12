from lib import *
from model import get_model
from helper_function import Helper

helper = Helper()


class CLIP_Processor:
    def __init__(self, test_query):
        self.dataset_dict, self.image_filenames = helper.dataset_dict, helper.image_filenames
        self.index = faiss.IndexFlatIP(512)
        self.test_query = test_query
        self.model, self.processor = get_model("CLIP")

    def get_image_embedding(self, image):
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)

        outputs = self.model.get_image_features(**inputs)
        
        return outputs.cpu().detach().numpy()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        
        outputs = self.model.get_text_features(**inputs)

        return outputs.cpu().detach().numpy()


    def embedding_image_database(self): # => store embedding of image in faiss
        SAVE_INTERVAL = 100

        os.makedirs("vector_db", exist_ok=True)
        path_index_json = Path("vector_db/index_CLIP.json")
        path_index_bin = Path("vector_db/index_CLIP.bin")

        if not path_index_bin.exists():
            for i, file in tqdm.tqdm(enumerate(self.image_filenames)):
                image = Image.open(file).convert("RGB")
                embedding = self.get_image_embedding(image)
                self.index.add(embedding)

                if i % SAVE_INTERVAL == 0:
                    faiss.write_index(self.index, "vector_db/index_CLIP.bin")
                faiss.write_index(self.index, "vector_db/index_CLIP.bin")
        else:
            self.index = faiss.read_index("vector_db/index_CLIP.bin")

        if not path_index_json.exists():
            with open("vector_db/index_CLIP.json", "w") as f:
                json.dump(self.image_filenames, f)


    def Query(self, prompt, top_k=5):

        if isinstance(prompt, str):
            embedding = self.get_text_embedding(prompt)
        else:
            embedding = self.get_image_embedding(prompt)

        distance_euclide, indices = self.index.search(embedding, top_k)
        # print("Distance:", distance_euclide) # đã ranking
        # print("Indices:", indices)

        return [self.image_filenames[i] for i in indices[0]]


    def run (self):

        # embedding image
        self.embedding_image_database()
 
        # Đo lượng RAM sử dụng trước khi inference
        ram_before_infer = psutil.virtual_memory().used / (1024 ** 2)  # MB
        # query image
        path_images = self.Query(self.image_query, top_k=5) # truy van anh
        # Đo lượng RAM sử dụng sau khi inference
        ram_after_infer = psutil.virtual_memory().used / (1024 ** 2)  # MB

        # plot image after query
        helper.plot_results(path_images)

        print(f"RAM Used by Model CLIP to Inference: {ram_after_infer - ram_before_infer:.2f} MB")

if __name__ == "__main__":
    dataset_dir = "test_query"
    test_querys = [os.path.join(dataset_dir, path_query) for path_query in os.listdir(dataset_dir)]
    test_query = Image.open(test_querys[1])

    # Show image query
    plt.imshow(test_query)
    plt.axis('off')
    plt.show()

    processor = CLIP_Processor(test_query=test_query)
    processor.run()