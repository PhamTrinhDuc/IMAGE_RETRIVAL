from lib import *
from model import get_model
from helper_function import Helper

helper = Helper()


class ViT_Processor:
    def __init__(self, image_query):
        self.dataset_dict, self.image_filenames = helper.dataset_dict, helper.image_filenames
        self.index = faiss.IndexFlatIP(768)
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
        path_index_json = Path("vector_db/index_ViT.json")
        path_index_bin = Path("vector_db/index_ViT.bin")

        if not path_index_bin.exists():
            for i, file in tqdm.tqdm(enumerate(self.image_filenames)):
                image = Image.open(file).convert("RGB")
                embedding = self.get_image_embedding(image)
                self.index.add(embedding)

                if i % SAVE_INTERVAL == 0:
                    faiss.write_index(self.index, "vector_db/index_ViT.bin")
                faiss.write_index(self.index, "vector_db/index_ViT.bin")
        else:
            self.index = faiss.read_index("vector_db/index_ViT.bin")

        if not path_index_json.exists():
            with open("vector_db/.json", "w") as f:
                json.dump(self.image_filenames, f)


    def Query(self, image_query: str, top_k: int): # => implement query
        query_embedding = self.get_image_embedding(image_query)
        
        distance_euclide, indices = self.index.search(query_embedding, top_k)
        # print("Distance:", distance_euclide) # đã ranking
        # print("Indices:", indices)

        path_images = [self.image_filenames[i] for i in indices[0]]
        return path_images


    def run (self):
        # embedding image
        self.embedding_image_database()
        # query image
        path_images = self.Query(self.image_query, top_k=6) # truy van anh
        # plot image after query
        helper.plot_results(path_images)

        # grid_image = helper.create_image_grid(path_images)

        # return grid_image


if __name__ == "__main__":
    dataset_dir = "test_query"
    test_query = [os.path.join(dataset_dir, path_query) for path_query in os.listdir(dataset_dir)]
    image_query = Image.open(test_query[0])

    # Show image query
    plt.imshow(image_query)
    plt.axis('off')
    plt.show()
    processor = ViT_Processor(image_query=image_query)
    processor.run()