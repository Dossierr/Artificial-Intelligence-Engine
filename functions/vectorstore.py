
from langchain.document_loaders import S3DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
import environ
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
import os
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


#loading ENV variables
env = environ.Env()
environ.Env.read_env()

#Log into bedrock as a client
client_boto3 = boto3.client(service_name="bedrock-runtime", 
                       aws_access_key_id=env('BEDROCK_AWS_ACCESS_KEY_ID'), 
                       aws_secret_access_key=env('BEDROCK_AWS_SECRET_KEY'))

#Defining the Model we want to use
modelId = "amazon.titan-embed-text-v1"         
bedrock_embedding = BedrockEmbeddings(model_id=modelId, 
                                      client=client_boto3)

env = environ.Env()
environ.Env.read_env()

def index_files(case_id, PERSIST):
    print("creating new index for: ", case_id)
    # No folder exists or we want to re index
    s3_folder_path = 'cases/'+str(case_id)
    # Loading the case files from S3
    loader = S3DirectoryLoader("dossierr", prefix=s3_folder_path, aws_access_key_id=env('S3_AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=env('S3_AWS_SECRET_KEY'))
    documents = loader.load()
    #Splitting the documents in smaller sizes
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # Storing in index
    database = Chroma.from_documents(docs, bedrock_embedding, persist_directory="./chroma_db/"+str(case_id))

    #index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist/"+case_id}).from_loaders([loader])
    return database

def chroma_index(case_id, query):
    PERSIST = True
    if PERSIST and os.path.exists("chroma_db/"+case_id):
        # So we have a folder with index and want to use it
        print("Reusing index:  "+case_id+"\n")
        #ectorstore = chroma(persist_directory="persist/"+case_id, embedding_function=bedrock_embedding())
        #index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        database = Chroma(persist_directory="./chroma_db/"+str(case_id), embedding_function=bedrock_embedding)

    else:
        print("No index found")
        database = index_files(case_id, PERSIST)
    
    return database