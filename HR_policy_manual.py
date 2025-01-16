import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
 
# Load Environment Variables
load_dotenv()
 
# API Keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
 
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
# File Paths
PDF_PATH = "HR-Handbook.pdf"
CHROMA_DB_PATH = "./chroma_db"
 
# Prompts
system_prompt = (
    "You are an AI assistant specialized in answering questions about HR policies. "
    "Provide clear, concise answers based on the provided context. "
    "If the information is not available, respond with: 'I'm sorry, I don't have that information right now.'"
    "Context:\n{context}\n\n"
)
 
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
 
# Helper function to create and load the vector database
def load_or_create_chroma_db():
    if not os.path.exists(CHROMA_DB_PATH):
        st.info("Creating vector database from PDF...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs)
        
        vectordb = Chroma.from_documents(
            documents=final_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
 
        return vectordb
    
    else:
        st.info("Loading vector database...")
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
 
# Load or create vector database
vectordb = load_or_create_chroma_db()
 
# Create retriever
retriever = vectordb.as_retriever()
 
# History-aware retriever
contextualize_q_system_prompt = (
    "You are tasked with improving user queries to ensure they are clear and self-contained. "
    "Use chat history to make ambiguous queries precise. If the query is already clear, return it as is."
)
 
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
 
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
 
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
 
# Conversatoiinal RAG chain
 
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
 
# Initialize Chat Message History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {'role': "assistant", 'content': "Hi, I'm your HR assistant! How can I help you today?"}
    ]
 
# Streamlit UI
st.markdown("<h1 style='text-align: center;'>HR Assistant</h1>", unsafe_allow_html=True)
 
# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
 
if prompt := st.chat_input(placeholder="Ask me anything about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
 
    # Generate response using RAG Chain
    response = conversational_rag_chain.invoke({"input": prompt}, config={"configurable": {"session_id": "hr_chat_session"}})
    response_text = response["answer"]
 
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)