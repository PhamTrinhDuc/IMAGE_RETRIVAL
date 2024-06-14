"""
LlamaParse là 1 dịch vụ phân tích PDF, ... hiệu quả
Llamaparse cung cấp API miễn phí tuy nhiên sẽ bị giới hạn:
+ 1K page 1 ngày free
+ Hết free: 0.003$ 1 page
"""
import os
from Module_RAG.config_app import config
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import joblib

config_app = config.get_config()

# Helper function to load and parse the input data
def load_or_parse_data():
    data_file = "Module_RAG/data/parsed_data.pkl"

    if os.path.exists(data_file):
        parsed_data = joblib.load(data_file)
    else:
        parsing_instruction = """
        Tài liệu được cung cấp là thông tin các sản phẩm của chúng tôi.
        Tài liệu gồm 2 phần: Các loại sản phẩm và thông tin sản phẩm.
        Mỗi sản phẩm chứa các thông tin như ID, tên sản phẩm, giá bán, mô tả sản phẩm, số lượng...
        """

        parser = LlamaParse(api_key=config_app["LlamaParse_API"],
                            result_type="markdown",
                            parsing_instruction=parsing_instruction,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data("Module_RAG/data/pdf_product.pdf")


        # Save the parsed data to a file
        # print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


# Helper function to load chunks into vectorstore.

def create_vector_database():

    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    print(llama_parse_documents[0].text[:100])

    if not os.path.exists("Module_RAG/data/output.md"):
        with open('Module_RAG/data/output.md', 'a') as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')
        
    markdown_path = "Module_RAG/data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    loader = DirectoryLoader("Module_RAG/data/", "**/*.md", show_progress=True)
    documents = loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create and persist a Chroma vector DB 
    os.makedirs("Module_RAG/vector_db", exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embed_model,
        persist_directory="Module_RAG/vector_db",
        collection_name="rag"
    )

    # print("Vector DB created successfully !")
    return vector_store, embed_model

# if __name__ == "__main__":
#     vector_store, embed_model = create_vector_database()