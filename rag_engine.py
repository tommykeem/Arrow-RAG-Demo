# Standard library imports
import os 
from pathlib import Path
 
 
# LLM and embedding model imports
from langchain_ollama.llms import OllamaLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA  
from langchain_huggingface import HuggingFaceEmbeddings

# Document loading and processing imports
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import WebBaseLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter

# Vector store and QA chain imports
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Verify NVIDIA API key is set
if os.getenv('NVIDIA_API_KEY') is None:
    raise ValueError('NVIDIA_API_KEY environment variable is not set')

# Initialize default LLM configuration for Ollama
llm = OllamaLLM(
    model="llama3.1:latest",
    temperature = 0
)

# Initialize default embedding model
embeddings = HuggingFaceEmbeddings()


def get_model(model, deployment):
    """Maps user-friendly model names to their corresponding model identifiers.
    
    Args:
        model (str): The selected model name
        deployment (str): The deployment type (DGX Local or NVIDIA NIM)
    
    Returns:
        str: The corresponding model identifier for the selected deployment
    """
    
    if deployment == 'DGX Local Deployment': 
        if model == 'Meta Llama 3.1' :
            return 'llama3.1:latest'
        elif model == 'Google Gemma 2':
            return 'gemma2:latest'
        elif model =='Microsoft Phi 3.5':
            return 'phi3.5:latest'
    if deployment == 'NVIDIA NIM':
        if model == 'Meta Llama 3.1' :
            return 'meta/llama-3.1-8b-instruct'
        elif model == 'Google Gemma 2':
            return 'google/gemma-2-9b-it'
        elif model =='Microsoft Phi 3.5':
            return 'microsoft/phi-3.5-moe-instruct'
    
def local_get_answer_upload_pdf(model, temperature, file_name, query):
    """Process a single uploaded PDF file using local deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        file_name (str): Name of the uploaded PDF file
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    llm = OllamaLLM(
        model=model,
        temperature = temperature
        )

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    
    file_path = f"{working_dir}/{file_name}"
    
    # loading the document
    loader = UnstructuredLoader(file_path)
    documents  = loader.load()
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(documents)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
        
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def local_get_answer_url(model, temperature, query):
    """Process multiple URLs specified here using local deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    llm = OllamaLLM(
        model=model,
        temperature = temperature
        )

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    
    # List of URLs to load documents from
    urls = [
       "https://en.wikipedia.org/wiki/Arrow_Electronics",
       "https://www.linkedin.com/company/arrow-electronics/posts/?feedView=all",
       "https://www.builtincolorado.com/company/arrow-electronics-inc",
       "https://www.bloomberg.com/profile/company/ARW:US"
    ]
    
    os.write(1,f"{urls}\n".encode())
    


    
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]

    docs_list = [item for sublist in docs for item in sublist]

    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(docs_list)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    

    return response["result"]

def local_get_answer_folder_pdf(model, temperature, query):
    """Process all PDFs in the docs folder using local deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    llm = OllamaLLM(
        model=model,
        temperature = temperature
        )

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    
    # Define the folder path
    folder_path = Path(f"{working_dir}/docs")

    # List all PDF files in the folder
    pdf_files = [file for file in folder_path.glob('*.pdf')]

    # loading the document
    docs = UnstructuredLoader(pdf_files).load()
  
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(docs)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
        
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]


      
    llm = OllamaLLM(
        model=model,
        temperature = temperature
        )

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    # Define the folder path
    folder_path = Path(f"{working_dir}/excel")


    # loading the document
    excel_files = list(folder_path.glob('*.xlsx'))
    
    all_docs = []
    for file in excel_files:
        docs = UnstructuredExcelLoader(str(file), mode="elements").load()
        all_docs.extend(docs)
    
    
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 00)
    
    text_chunks = text_splitter.split_documents(docs)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
        
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def nim_get_answer_folder_pdf(model, temperature, query):
    """Process all PDFs in the docs folder using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    folder_path = Path(f"{working_dir}/docs")
    
    api_key = NVIDIA_API_KEY
    
    llm = ChatNVIDIA(
        model=model,
        api_key=api_key, 
        temperature=temperature,
        top_p=0.7,
        max_tokens=1024,
    )
    
    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    os.write(1,f"{api_key}\n".encode())
    
    # List all PDF files in the folder
    pdf_files = [file for file in folder_path.glob('*.pdf')]

    # loading the document
    docs = UnstructuredLoader(pdf_files).load()
  
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(docs)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def nim_get_answer_url(model, temperature, query):
    """Process multiple URLs specified here using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    api_key = NVIDIA_API_KEY
    
    llm = ChatNVIDIA(
        model=model,
        api_key=api_key, 
        temperature=temperature,
        top_p=0.7,
        max_tokens=1024,
    )
    
    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    os.write(1,f"{api_key}\n".encode())

    # List of URLs to load documents from
    urls = [
        "https://en.wikipedia.org/wiki/Arrow_Electronics",
       "https://www.linkedin.com/company/arrow-electronics/posts/?feedView=all",
       "https://www.builtincolorado.com/company/arrow-electronics-inc",
       "https://www.bloomberg.com/profile/company/ARW:US"
    ]
    
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    
    text_chunks = text_splitter.split_documents(docs_list)

    # vector embeddings from text chunks 
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def nim_get_answer_upload_pdf(model, temperature, file_name, query):
    """Process a single uploaded PDF file using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        file_name (str): Name of the uploaded PDF file
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    api_key = NVIDIA_API_KEY
    
    llm = ChatNVIDIA(
        model=model,
        api_key=api_key, 
        temperature=temperature,
        top_p=0.7,
        max_tokens=1024,
    )

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())
    os.write(1,f"{api_key}\n".encode())
    
    file_path = f"{working_dir}/{file_name}"
    
    # loading the document
    loader = UnstructuredLoader(file_path)
    documents = loader.load()
    
    # create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    
    text_chunks = text_splitter.split_documents(documents)

    # vector embeddings from text chunks 
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]