import streamlit as st
from gtts import gTTS
from IPython.display import Audio
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the OpenAI embeddings
openAI_embeddings = OpenAIEmbeddings()

# Function to get vectorstore from a document
def get_vectorstore_from_doc(url):
    loader = PyPDFLoader(url)
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(document_chunks, openAI_embeddings)
    return vector_store

vector_store = get_vectorstore_from_doc('your_document_url.pdf')

# Function to get the conversational retrieval chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", api_key=openai_api_key)
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

# Function to convert text to speech
def text_to_speech(text, lang='ar', tld='com'):
    tts = gTTS(text=text, lang=lang, slow=False, tld=tld)
    tts.save("response.mp3")
    return Audio("response.mp3", autoplay=True)

# Streamlit interface
st.title("Conversational AI with Streamlit")
st.write("Ask a question to the AI and get a spoken response:")

query = st.text_input("Enter your question:")

# Initialize vector store and conversational chain
# You need to replace 'your_document_url.pdf' with the actual URL or path of your document
conversation_chain = get_context_retriever_chain(vector_store)

# Function to get the answer from the conversational chain
def get_answer(query):
    result = conversation_chain({"question": query})
    answer = result["answer"]
    print(answer)
    return answer

if st.button("Get Answer"):
    if query:
        answer = get_answer(query)
        st.write("Answer:", answer)
        text_to_speech(answer)
        st.audio("response.mp3")
    else:
        st.write("Please enter a question.")
