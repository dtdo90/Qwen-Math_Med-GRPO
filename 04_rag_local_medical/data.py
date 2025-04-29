import os
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

def count_tokens(text, model="cl100k_base"):
    """ Count the number of tokens in the text using tiktoken"""
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))


def split_documents(document_path):
    """ Split document into smaller chunks to improve retrieval
            (1) Use RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
            (2) Ensure chunks are appropriately sized for embedding and retrieval
            (3) Count total chunks and total tokens
        Returns:
            List of split document objects
    """
    # load document
    loader=PyPDFLoader(document_path)
    pages=loader.load()

    # initialize tiktokken splitter
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=100
    )
    # split documents into chunks
    split_docs=text_splitter.split_documents(pages)
    print(f"Created {len(split_docs)} chunks from documents")

    # count total tokens
    total_tokens=0
    for doc in split_docs:
        total_tokens+= count_tokens(doc.page_content)
    print(f"Total tokens in split documents: {total_tokens}")

    return split_docs

def create_vectorstore(split_docs):
    """ Create vector store from the chunks using FAISS
        Return:
            FAISS: a vector store containing the embedded documents
    """
    print("Creating FAISS vector store ...")
    # initialize text embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # create vector store from documents using FAISS
    persist_path = os.path.join(os.getcwd(), "local_vectorstore")
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    print("FAISS vector store created successfully!")
    
    # save it
    vectorstore.save_local(persist_path)
    print("FAISS vector store was persisted to ", persist_path)

    return vectorstore

if __name__=="__main__":
    # get split documents from the pdf
    split_docs=split_documents("medical.pdf")
    # create a FAISS vector store
    vectorstore=create_vectorstore(split_docs)

    # example usage
    query="What are symptoms of flu?"
    MAX_DISTANCE = 1  # FAISS uses L2 distance, lower is better
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
    
    # get relevant documents
    relevant_docs=[]
    print("\nDocument distances (lower is better): ")
    for doc,score in docs_with_scores:
        if score<=MAX_DISTANCE:
            relevant_docs.append(doc)
            print(f"Distance: {score:.4f} - Document selected")
        else:
            print(f"Distance: {score:.4f} - Document too far")
    print(f"\nRetrieved {len(relevant_docs)} relevant documents")

    # print out the documents
    for i,d in enumerate(relevant_docs):
        print(f"\n------- Document {i+1} -------\n")
        print(d.metadata['source'])
        print(d.page_content)

