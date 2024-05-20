import streamlit as st
from gtts import gTTS
from IPython.display import Audio
#from playsound import playsound
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

# Set the OpenAI API key

load_dotenv()


# Initialize the components
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") 
text_splitter = CharacterTextSplitter()
embeddings = OpenAIEmbeddings()
# loader = PyPDFLoader("src1/3_TradeRecord_ar-MA.pdf")
# docs = loader.load()

# char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
# doc_texts = char_text_splitter.split_documents(docs)

# openAI_embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]) 
# #vStore = Chroma.from_documents(doc_texts, openAI_embeddings)

# vStore = Chroma.from_documents(doc_texts, openAI_embeddings, persist_directory = 'src1/')
# vStore.persist()

#vectordb = Chroma(embedding_function=embeddings)

persist_directory = '/'
openAI_embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=openAI_embeddings)

# Create the conversational chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    memory=memory
)

# Function to convert text to speech
def text_to_speech(text, lang='ar', tld='com.ar'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("response.mp3")
    return Audio("response.mp3", autoplay=True)

# Function to get the answer from the conversational chain
def get_answer(query):
    result = conversation_chain({"question": query})
    answer = result["answer"]
    print(answer)
    return answer

# Streamlit interface
st.title("Conversational AI with Streamlit")
st.write("Ask a question to the AI and get a spoken response:")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        answer = get_answer(query)
        st.write("Answer:", answer)
        audio = text_to_speech(answer)
        st.audio("response.mp3")
    else:
        st.write("Please enter a question.")

