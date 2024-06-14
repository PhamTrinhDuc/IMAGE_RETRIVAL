

from groq import Groq 
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from Module_RAG.config_app import config
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from Module_RAG.LlamaParse import create_vector_database

config_app = config.get_config()

def create_retriever():
    # Instantiate embedding model
    vector_store, embed_model = create_vector_database()

    # Instantiate Vectorstore
    vector_store = Chroma(
        embedding_function=embed_model,
        persist_directory="Module_RAG/vector_db",
        collection_name='rag'
    )

    retriever = vector_store.as_retriever(
        search_kwargs={'k': 3}
    )
    # print(retriever.invoke("Có bao nhiêu điều hòa ?")[0].page_content)
    return retriever

# Create a Custom Prompt Template

def set_custom_prompt():
    custom_prompt = """Sử dụng từng phần thông tin để trả lời câu hỏi từ người dùng.
    Nếu bạn không biết câu trả lời hãy nói "Có vẻ sản phẩm này không có trong kho hàng của chúng tôi." 
    Không tự tạo ra câu trả lời của riêng mình.

    Context: {context}
    Question: {question}

    Bạn chỉ được trả lời bằng tiếng việt và trả lời 1 cách chính xác nhất.
    """
    prompt = PromptTemplate(
        template=custom_prompt,
        input_variables=['content', 'question'],
    )
    return prompt


# Instantiate the Retrieval Question Answering Chain
def create_Chain_QA():
    llm = ChatGroq(
        model='Llama3-70b-8192',
        api_key=config_app["Groq_API"])
    prompt = set_custom_prompt()
    retriever = create_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})
    return qa


# if __name__ == "__main__":
#     qa = create_Chain_QA()
#     response = qa.invoke({"query": "Liệt kê 5 sản phẩm bếp từ ?"})
#     print(response['result'])