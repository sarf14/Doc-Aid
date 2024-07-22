import streamlit as st
import random
import asyncio
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

# Define the custom prompt template for the RetrievalQA chain
custom_prompt_template = """
Use the following pieces of information to answer the user's question in a comprehensive and informative way, leveraging the retrieved documents. If the retrieved documents are not helpful, try to provide relevant information or rephrase the question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Initialize the PromptTemplate with the custom prompt
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Function to create a RetrievalQA chain
def retrieval_qa_chain(llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),  # Retrieve top 5 documents
        return_source_documents=False,  # Do not return source documents
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Function to load the language model (LLM)
def load_llm():
    llm = ChatGroq(
        groq_api_key="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",
        model_name="Llama3-8b-8192",
        max_tokens=8192,
        temperature=0.5
    )
    return llm

# Function to create vector database from in-memory files
def create_vector_db_from_memory(file_list):
    documents = []
    
    for uploaded_file in file_list:
        file_bytes = uploaded_file.read()
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        text = ""
        
        # Convert file bytes to text based on file extension
        if file_extension == 'pdf':
            pdf = PdfReader(BytesIO(file_bytes))
            for page in pdf.pages:
                text += page.extract_text()
        elif file_extension == 'docx':
            doc = DocxDocument(BytesIO(file_bytes))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        elif file_extension == 'pptx':
            prs = Presentation(BytesIO(file_bytes))
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        elif file_extension == 'txt':
            text = file_bytes.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Create Document objects and append to the list
        documents.append(Document(page_content=text))

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Create FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Asynchronous function to initialize the QA bot
async def qa_bot():
    llm = load_llm()
    return llm

responses = [
    "Here's the information I found for you. If you need anything else, just let me know!",
    "I've got some details on that topic. Feel free to ask if you'd like more information or have other questions!",
    "I hope this answers your question. If you'd like more details or have any other inquiries, I'm here to help!",
    "Here are the details I have for you. If you have more questions or need further assistance, just ask!",
    "I've gathered the relevant information for you. Let me know if there's anything more you'd like to know or if you have any other questions!"
]

# Function to get the QA chain
def get_chain(vectorstore):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm = loop.run_until_complete(qa_bot())
    return retrieval_qa_chain(llm, vectorstore)

# Function to handle the QA chain invocation and context-based queries
def new_func(chain, question_text, context=None, max_length=4096):
    try:
        if context:
            combined_text = f"{context}\n{question_text}"
        else:
            combined_text = question_text

        # Ensure the combined text length does not exceed the max_length
        if len(combined_text) > max_length:
            # Truncate the context to fit within the max_length
            truncated_context = context[-(max_length - len(question_text)):]
            combined_text = f"{truncated_context}\n{question_text}"

        result = chain.invoke(combined_text)
        return result
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"

# Custom CSS for dark mode and styled components
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header img {
        height: 60px; /* Adjust height as needed */
        margin-right: 20px;
    }
    .header h1 {
        font-size: 3em;
        color: #4187CE;
        margin: 0;
    }
    .input-section {
        margin: 20px 0;
        padding: 20px;
        background-color: #2C2C2C;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .history div {
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .user {
        font-weight: bold;
        color: white;
        justify-content: flex-end;
    }
    .bot {
        font-weight: bold;
        color: white;
        justify-content: flex-start;
    }
    .message {
        max-width: 70%;
        padding: 10px;
        border-radius: 8px;
    }
    .user .message {
        background-color: #BD1362;
        color: #ffffff;
        align-self: flex-end;
    }
    .bot .message {
        background-color: #5FB233;
        color: #ffffff;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section with logo and title
st.image("./images/godrej.png", width=100)  # Adjust width as needed
st.markdown("""
    <div class="header">
        <h1>Godrej Guide Support Chat</h1>
    </div>
    """, unsafe_allow_html=True)

# Greet the user
st.write('Hello there! 👋 Welcome to the support chat. How can I assist you today?')

# Initialize session state for chat history and context
if 'history' not in st.session_state:
    st.session_state['history'] = []
    st.session_state['context'] = ""

# Create a form for user input and submission
with st.form(key='chat_form', clear_on_submit=True):
    # Text input for user's question
    user_input = st.text_input("Your Question:", key="input_field")
    # Submit button
    submit_button = st.form_submit_button(label='Submit')

    if submit_button and user_input:
        # Get the vector store from uploaded file
        vectorstore = st.session_state.get('vectorstore', None)
        
        if vectorstore is None:
            st.write("Please upload files to create the vector store first.")
        else:
            # Retrieve the chain
            chain = get_chain(vectorstore)
            
            # Invoke the chain to get the response
            response = new_func(chain, user_input, context=st.session_state['context'])
            
            # Update chat history and context
            st.session_state['history'].append({"user": user_input, "bot": response})
            st.session_state['context'] += f"\n{user_input}\n{response}"
            
            # Display the conversation history
            for chat in st.session_state['history']:
                st.markdown(f"<div class='history'><div class='user'><div class='message'>{chat['user']}</div></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='history'><div class='bot'><div class='message'>{random.choice(responses)}</div></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='history'><div class='bot'><div class='message'>{chat['bot']}</div></div></div>", unsafe_allow_html=True)

# File upload section for creating vector store
st.markdown("## Upload Files to Create Vector Store")
uploaded_files = st.file_uploader("Upload PDF, DOCX, PPTX, or TXT files", accept_multiple_files=True)

if st.button("Create Vector Store"):
    if uploaded_files:
        # Create vector store from uploaded files
        vectorstore = create_vector_db_from_memory(uploaded_files)
        st.session_state['vectorstore'] = vectorstore
        st.write("Vector store created successfully!")
    else:
        st.write("Please upload at least one file.")
