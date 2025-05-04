import re, os
import tiktoken
from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def count_tokens(text, model="cl100k_base"):
    """ Count the number of tokens in the text using tiktoken"""
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def bs4_extractor(html: str) -> str:
    """ Extract and clean content from HTML string,
    """
    # parse html: use BeautifulSoup with "lxml" parser to create structured object from raw html
    soup = BeautifulSoup(html, "lxml")

    # Content selection: multiple content selectors
    content_selectors = [
        "article.md-content__inner",  # article selector,  content within markdown-style documentation
        "div#main-content",           # Common content container
        "main",                       # Main content area
        "div.content",                # Generic content div
        "article"                     # Any article tag
    ]
    
    content = None
    for selector in content_selectors: # iterate over selectors
        main_content = soup.select_one(selector)
        if main_content:
            content = main_content.get_text()
            break
    
    # If no specific content found, use the whole document
    if not content:
        content = soup.text

    # Clean up the content
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    return content

def load_cancer_docs():
    """ Load documentation on cancer from multiple authoritative sources
            (1) Use RecursiveUrlLoader to fetch pages 
            (2) Count total documents and tokens loaded

        Returns 
            list1: list of document objects containing loaded content
            list2: list of tokens per document
    """
    print("Loading cancer documentation ...")
    urls = [
        # MedlinePlus - General Cancer Information
        "https://medlineplus.gov/cancer.html",
        "https://medlineplus.gov/cancerprevention.html",
        "https://medlineplus.gov/cancertreatment.html",
        
        # MedlinePlus - Specific Cancers
        "https://medlineplus.gov/lungcancer.html",
        "https://medlineplus.gov/breastcancer.html",
        "https://medlineplus.gov/colorectalcancer.html",
        "https://medlineplus.gov/prostatecancer.html",
        "https://medlineplus.gov/skincancer.html",
        
        # MedlinePlus - Cancer Tests
        "https://medlineplus.gov/lab-tests/lung-cancer-genetic-tests/",
        "https://medlineplus.gov/lab-tests/skin-cancer-screening/",
        "https://medlineplus.gov/lab-tests/ca-19-9-blood-test-pancreatic-cancer/",
        "https://medlineplus.gov/lab-tests/ca-125-blood-test-ovarian-cancer/",
        "https://medlineplus.gov/lab-tests/colorectal-cancer-screening-tests/",
        
        # CDC Cancer Information
        "https://www.cdc.gov/cancer/index.htm",
        "https://www.cdc.gov/cancer/dcpc/prevention/index.htm",
        
        # National Cancer Institute
        "https://www.cancer.gov/about-cancer/understanding/what-is-cancer",
        "https://www.cancer.gov/about-cancer/causes-prevention",
        "https://www.cancer.gov/about-cancer/diagnosis-staging"
    ]
    
    docs = []
    for url in urls:
        try:
            loader = RecursiveUrlLoader(
                url,
                max_depth=3,  # Reduced depth but more URLs
                extractor=bs4_extractor,
                prevent_outside=True  # Only follow links within the same domain
            )
            # load documents with lazy loading (memory efficient)
            docs_lazy = loader.lazy_load()
            # load documents and track urls
            for d in docs_lazy:
                docs.append(d)
        except Exception as e:
            print(f"Error loading {url}: {str(e)}")
            continue
            
    print(f"Loaded {len(docs)} documents from cancer documentation")
    print("\nLoaded URLs:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source','Unknown URL')}")
    
    # count total tokens in documents
    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        total_tokens += count_tokens(doc.page_content)
        tokens_per_doc.append(count_tokens(doc.page_content))
    print(f"Total tokens in loaded documents: {total_tokens}")

    return docs, tokens_per_doc

def save_cancer_full(documents):
    """ Save documents to a file"""
    # open output file
    output_filename = "cancer_doc.txt"
    with open(output_filename, "w") as f:
        # write each document
        for i, doc in enumerate(documents):
            # get url from metadata
            source = doc.metadata.get('source', 'Unknown URL')
            # write the doc with proper formatting
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")

    print(f"Documents concatenated into {output_filename}")

def split_documents(documents):
    """ Split documents into smaller chunks to improve retrieval
            (1) Use RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
            (2) Ensure chunks are appropriately sized for embedding and retrieval
            (3) Count total chunks and total tokens
        Returns:
            list of split document objects
    """
    # initialize with tiktoken splitter for accurate token counting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=100
    )
    # split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from documents")

    # Count total tokens in split documents
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    print(f"Total tokens in split documents: {total_tokens}")
    
    return split_docs

def create_vectorstore(splits):
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
    persist_path = os.getcwd() + "/cancer_vectorstore"
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    print("FAISS vector store created successfully!")
    
    # save it
    vectorstore.save_local(persist_path)
    print("FAISS vector store was persisted to ", persist_path)

    return vectorstore

if __name__ == "__main__":
    documents, tokens_per_doc = load_cancer_docs()
    # save documents to a file
    save_cancer_full(documents)
    # split doc
    split_docs = split_documents(documents)

    # create vector store and save it
    vectorstore = create_vectorstore(split_docs)

    # get relevant documents for the query with similarity threshold
    query = "What is lung cancer?"
    MAX_DISTANCE = 1  # Maximum acceptable L2 distance
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
    
    relevant_docs = []
    print("\nDocument distances (lower is better):")
    for doc, score in docs_with_scores:
        if score <= MAX_DISTANCE:
            relevant_docs.append(doc)
            print(f"Distance: {score:.4f} - Document selected")
        else:
            print(f"Distance: {score:.4f} - Document too far")
    
    print(f"\nRetrieved {len(relevant_docs)} relevant documents")

    for i, d in enumerate(relevant_docs):
        print(f"\n------- Document {i+1} --------\n")
        print(d.metadata['source'])
        print(d.page_content)
