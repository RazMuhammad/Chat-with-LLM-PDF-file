from groq import Groq
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st


class simple_chat:
    def __init__(self, prompt):
        self.prompt = prompt
        self.key = st.secrets['GROQ_API_KEY']
        self.client = Groq(api_key=self.key)
        


    def chat(self):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a General type of assistant answering questions from all domains.",
                },
                {
                    "role": "user",
                    "content": self.prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content



class RAG:
    def __init__(self, file_name, query):
        self.file_name = file_name
        self.query = query
        self.key = st.secrets['GROQ_API_KEY']
        self.client = Groq(api_key=self.key)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def extract_text_from_pdf(self):
        text = ""
        with fitz.open(self.file_name) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def split_text_with_langchain(self, text, chunk_size=300, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def embed_text_chunks(self, chunks):
        embeddings = self.embedding_model.encode(chunks)
        return np.array(embeddings)

    def retrieve_relevant_chunks(self, query, chunks, embeddings, k=3):
        query_embedding = self.embedding_model.encode([query])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        distances, indices = index.search(query_embedding, k)
        return [chunks[idx] for idx in indices[0]]

    def generate_answer(self, relevant_chunks):
        chat_content = " ".join(relevant_chunks)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a RAG base specific assistant.",
                },
                {
                    "role": "user",
                    "content": chat_content,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

    def run(self):
        pdf_text = self.extract_text_from_pdf()
        text_chunks = self.split_text_with_langchain(pdf_text)
        embeddings = self.embed_text_chunks(text_chunks)
        relevant_chunks = self.retrieve_relevant_chunks(self.query, text_chunks, embeddings)
        return self.generate_answer(relevant_chunks)



st.title("Chat with LLM || PDF file")
st.write("Interact with an AI assistant for general queries or upload a PDF to ask specific questions about its content. This app seamlessly combines chat and PDF query capabilities for a versatile user experience.")

# Initialize client (replace with Groq if needed)
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Check for uploaded PDF and store it
uploaded_file = st.file_uploader("Upload a PDF file for question about its contents", type=['pdf'])

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question or enter a prompt"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    if uploaded_file:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        rag_obj = RAG("uploaded_file.pdf", user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing PDF..."):
                response = rag_obj.run()
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
    else:
        chat_obj = simple_chat(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_obj.chat()
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
