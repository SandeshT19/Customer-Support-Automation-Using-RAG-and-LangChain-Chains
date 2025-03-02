import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the file and directories
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "apple_iphone_11_reviews.csv")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Embedding initialization with HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_folder="Local_Folder/models/cache/models--BAAI--bge-small-en-v1.5")

def check_file_existence(file_path):
    """Check if a file exists."""
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist. Please check the path.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def initialize_vector_store():
    """Initialize vector store if not already initialized."""
    if not os.path.exists(persistent_directory):
        logger.info("Persistent directory does not exist. Initializing vector store...")
        check_file_existence(file_path)

        # Load CSV file and split documents
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()

        # Split documents into chunks
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(documents)

        logger.info(f"Number of document chunks: {len(docs)}")
        logger.info(f"Sample chunk: {docs[0].page_content[:200]}")

        # Create vector store
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        logger.info("Vector store initialized successfully.")
    else:
        logger.info("Vector store already exists. Skipping initialization.")

def retrieve_documents(query):
    """Retrieve relevant documents based on query."""
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 7, "score_threshold": 0.35})
    relevant_docs = retriever.invoke(query)
   
    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Relevant results document {i}:\n{doc.page_content}\n")
    logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")
    return relevant_docs

def generate_response(query, relevant_docs,model):
    """Generate response based on the query and relevant documents."""
    combined_input = f"Here are some documents that might help answer the question: {query}\n\nRelevant Documents:\n" + "\n\n".join([doc.page_content for doc in relevant_docs]) + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."



    # Define the messages for the model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": combined_input},
    ]

    response = model.invoke(messages)
    logger.info(f"Generated response: {response}")
    return response

def classify_feedback(feedback,model):
    """Classify sentiment of feedback and generate appropriate response."""
    classification_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ])
    

# Define prompt templates for different feedback types
    positive_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human",
            "Generate a thank you note for this positive feedback: {feedback}."),
        ]
    )

    negative_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human",
            "Generate a response addressing this negative feedback: {feedback}."),
        ]
    )

    neutral_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a request for more details for this neutral feedback: {feedback}.",
            ),
        ]
    )

    escalate_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a message to escalate this feedback to a human agent: {feedback}.",
            ),
        ]
    )
    branches = RunnableBranch(
        (lambda x: "positive" in x, positive_feedback_template | model | StrOutputParser()),
        (lambda x: "negative" in x, negative_feedback_template | model | StrOutputParser()),
        (lambda x: "neutral" in x, neutral_feedback_template | model | StrOutputParser()),
        escalate_feedback_template | model | StrOutputParser(),
    )

    classification_chain = classification_template | model | StrOutputParser()
    chain = classification_chain | branches

    result = chain.invoke({"feedback": feedback})
    return result

def main():
    """Main function to run the entire process."""
    query = "find the top bad review _ text for review rating 1 out of 5?"
    model = LlamaCpp(
        model_path="Local_Folder/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=1,
        n_batch=8,
        n_ctx=6000,
        f16_kv=True,
        n_threads=4,
        verbose=False,
    )
    try:
        initialize_vector_store()
        
        # Retrieve relevant documents
        relevant_docs = retrieve_documents(query)
        
        # Generate response based on retrieved documents
        generated_response = generate_response(query, relevant_docs,model)
        #printing the response geneated by RAG and Modle
        print(f"Generated Response from RAG : \n {generated_response}")
        # Classify feedback
        feedback = generated_response  # Assuming the generated response is feedback
        feedback_result = classify_feedback(feedback,model)
        
        logger.info("Feedback classification result:")
        logger.info(feedback_result)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
