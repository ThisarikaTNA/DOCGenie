import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with Your PDF Document", page_icon=":robot_face:")
    st.header("Chat with Your PDF Document using DOCGenie")

    # Sidebar configuration
    st.sidebar.header("Tired of digging through manuals or bothering colleagues for simple questions?")
    st.sidebar.write(
        "Our intelligent chatbot is your new best friend! It's powered by your uploaded PDF document – the ultimate guide to how we do things. This means you have a pocket-sized expert always ready to answer your questions. Want to know how to run a specific report? Just ask the chatbot! It's like having a super-efficient, always-available assistant at your fingertips.")

    # File uploader for the PDF document
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        if 'sop_processed' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
            raw_text = get_pdf_text([uploaded_file])
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state['sop_processed'] = True
            st.session_state['current_file'] = uploaded_file.name

        user_question = st.text_input("Ask a Question from the uploaded PDF Document", key='user_question', value="")

        if st.button("Submit"):
            answer = user_input(user_question)
            st.write("Reply:", answer)
            st.button("Ask another question")

        if st.button("Clear"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
