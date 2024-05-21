import streamlit as st
# from gtts import gTTS
# from IPython.display import Audio
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
# from gtts import gTTS
# from IPython.display import Audio
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()


# Initialize the OpenAI embeddings
openAI_embeddings = OpenAIEmbeddings()

# Function to get vectorstore from a document
def get_vectorstore_from_doc(pdf):
    # loader = PyPDFLoader(url)
    # document = loader.load()
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = text_splitter.split_text(text)
    # document_chunks = text_splitter.split_documents(document)
    document_chunks = text_splitter.split_text(text)
    
    # vector_store = Chroma.from_documents(document_chunks, openAI_embeddings)
    vector_store = Chroma(embedding_function=openAI_embeddings)
    vector_store.add_texts(document_chunks)
    return vector_store

vector_store = get_vectorstore_from_doc('docs/3_TradeRecord_ar-MA.pdf')

# Function to get the conversational retrieval chain
def get_context_retriever_chain(vector):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    retriever = vector.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

# Function to convert text to speech
# def text_to_speech(text, lang='ar', tld='com'):
#     tts = gTTS(text=text, lang=lang, slow=False, tld=tld)
#     tts.save("response.mp3")
#     return Audio("response.mp3", autoplay=True)

# Streamlit interface
st.set_page_config(page_title="Smartify Law Assistant Bot", page_icon="⚖️", layout="wide")
#st.title("Welcome To Smartify Smart Law Assistant bot")
st.title("Almohami: Your Intelligent Legal Companion is Here for your assistance.")

# st.write("This is a beta version of our Assistant, focus your question on Commercial Code of 🇲🇦 please !!")
st.markdown("**This is a beta version of our Assistant, focus your question on Commercial Code of 🇲🇦 please !!** - ** !! هذه نسخة تجريبية من مساعدنا، يرجى تركيز سؤالك على القانون التجاري للمغرب 🇲🇦 من فضلك **")

# st.write("Please Enter Your Question (Arabic input) ")

st.sidebar.image("docs/smartify_bot.PNG", use_column_width = True)
query = st.text_input("Enter Your Question: :** - ** :الرجاء إدخال سؤالك **")

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
        # text_to_speech(answer)
        # st.audio("response.mp3")
    else:
        st.write("Please enter a question.")
