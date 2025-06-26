import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
#from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load the Documents
def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        #elif file_path.endswith(('.docx','.doc')):
           # loader = UnstructuredWordDocumentLoader(file_path)
        else:
            loader = UnstructuredLoader(file_path)
        documents.extend(loader.load())

    return documents

# Document Processing
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=20)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

# Initialize the Groq model
def model_init():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2
    )
    return llm

# Create the RetrievalQA chain
def create_qa_chain(llm,vector_store):
    prompt = """You are a helpful AI assistant who answers questions based on the provided context.
    Your task is to provide accurate answers to the user's questions.
    if the answer is not present in the context, You can take a guess based on the context and provide a reasonable answer. 
    Dont forget to mention that you are guessing.
    Use the following pieces of context to answer the question.
    {context}
    Question: {question}
    Answer:"""
    prompt_template = PromptTemplate(template=prompt, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs={"prompt": prompt_template},
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Main function to run the application
def main():
    #file_paths = [r'C:\Users\Sraavya\OneDrive\Desktop\personal data\Sulakshana_Sraavya_Resume.pdf']
    file_paths = ['sample_pdf.pdf', 'sample_txt.txt']
    documents = load_documents(file_paths)

    vector_store = process_documents(documents)

    llm = model_init()

    qa_chain = create_qa_chain(llm, vector_store)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        result = qa_chain.invoke({"query": question})
        print("\nAnswer:", result['result'])
        for doc in result["source_documents"]:
            print(doc.metadata.get("source", "Unknown Source"), "- Page", doc.metadata.get("page", "N/A"))


if __name__ == "__main__":
    main()
