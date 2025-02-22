import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langsmith import Client
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import uuid

# Load environment variables
load_dotenv()

# Initialize Pinecone client globally
pc = None

def init_langsmith():
    """Initialize LangSmith client with API key from environment variables."""
    try:
        # Set LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ollama-chat-app")
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        
        # Initialize LangSmith client
        client = Client()
        print("Successfully initialized LangSmith client")
        
        # List existing projects
        projects = client.list_projects()
        print(f"Available projects: {[project.name for project in projects]}")
        
        return True
    except Exception as e:
        print(f"Error initializing LangSmith: {str(e)}")
        return False

def init_pinecone():
    """Initialize Pinecone client with API key from environment variables."""
    try:
        global pc
        # Initialize Pinecone with API key
        pc = Pinecone(api_key=os.getenv('PINECODE_API_KEY'))
        print("Successfully initialized Pinecone client")
        
        # List existing indexes
        indexes = pc.list_indexes()
        print(f"Active indexes: {indexes.names()}")
        
        return True
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return False

def process_pdfs():
    """Process PDFs and generate embeddings."""
    try:
        global pc
        if pc is None:
            raise Exception("Pinecone client not initialized")
            
        # Initialize embeddings model
        embeddings = OpenAIEmbeddings()
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create Pinecone index if it doesn't exist
        index_name = "pdf-embeddings"
        dimension = 1536  # OpenAI embeddings dimension
        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Created new Pinecone index: {index_name}")
        
        # Get the index
        index = pc.Index(index_name)
        
        # Process each PDF in the pdfs directory
        pdf_dir = "pdfs"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            print(f"Created directory: {pdf_dir}")
            return
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                print(f"\nProcessing: {filename}")
                
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Split text into chunks
                docs = text_splitter.split_documents(pages)
                print(f"Split into {len(docs)} chunks")
                
                # Generate embeddings and upload to Pinecone
                for i, doc in enumerate(docs):
                    # Generate embedding
                    embedding = embeddings.embed_query(doc.page_content)
                    
                    # Create metadata
                    metadata = {
                        "text": doc.page_content,
                        "source": filename,
                        "page": doc.metadata.get("page", 0)
                    }
                    
                    # Upload to Pinecone
                    index.upsert(
                        vectors=[(str(uuid.uuid4()), embedding, metadata)]
                    )
                    
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(docs)} chunks")
                
                print(f"Completed processing: {filename}")
        
        print("\nFinished processing all PDFs")
        return True
    
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        return False

if __name__ == "__main__":
    # Test both initializations
    print("\nInitializing LangSmith...")
    if init_langsmith():
        print("\nInitializing Pinecone...")
        if init_pinecone():
            print("\nProcessing PDFs...")
            process_pdfs()
        else:
            print("Skipping PDF processing due to Pinecone initialization failure")
    else:
        print("Skipping remaining steps due to LangSmith initialization failure")
