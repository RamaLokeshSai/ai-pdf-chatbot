import streamlit as st
import fitz
import google.generativeai as genai
import os
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

#session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# CONFIG


st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)



# LOAD ENVIRONMENT VARIABLES

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)



# LOAD MODELS


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_gemini_model():

    return genai.GenerativeModel(
        "gemini-2.5-flash"
    )


embedding_model = load_embedding_model()
gemini_model = load_gemini_model()



# CHUNKING FUNCTION


def create_chunks(
    text,
    chunk_size=1000
):

    paragraphs = text.split("\n\n")

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) < chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            chunks.append(current_chunk)

            current_chunk = paragraph + "\n\n"

    if current_chunk:

        chunks.append(current_chunk)

    return chunks



# TITLE


st.title("📚 AI PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about it."
)



# FILE UPLOADER


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)



# PROCESS PDF


if uploaded_file:

    # Read PDF
    pdf_bytes = uploaded_file.read()

    # Open PDF
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(doc)

    # Extract text
    text = ""

    for page in doc:

        text += page.get_text()


    # Chunking
    chunks = create_chunks(text)

    st.write(
        "Number of Chunks:",
        len(chunks)
    )

    # Generate Embeddings
    # with st.spinner(
    #     "Generating embeddings..."
    # ):

    if st.session_state.pdf_name != uploaded_file.name:

        chunks = create_chunks(text)

        chunk_embeddings = embedding_model.encode(
            chunks,
            batch_size=32,
            convert_to_tensor=True
        )

        st.session_state.chunks = chunks
        st.session_state.embeddings = chunk_embeddings
        st.session_state.pdf_name = uploaded_file.name

    else:

        chunks = st.session_state.chunks
        chunk_embeddings = st.session_state.embeddings

    #FAISS INDEX
    embeddings_np = chunk_embeddings.cpu().numpy()

    dimension = embeddings_np.shape[1]

    if st.session_state.faiss_index is None:

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings_np)

        st.session_state.faiss_index = index

    else:

        index = st.session_state.faiss_index
    #st.success("FAISS Index Created")


    #sidebar

    with st.sidebar:

        st.title("📚 AI PDF Chatbot")

        st.success("PDF Loaded")

        st.write(
            f"📄 {uploaded_file.name}"
        )

        st.write(
            f"🧩 Chunks: {len(chunks)}"
        )
        st.write(
            f"📖 Total-Pages: {total_pages}"
        )


        #Clear chat button

        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []

            st.rerun()

        # st.write(
        #     f"🧠 Embeddings: {len(chunk_embeddings)}"
        # )
        # col1, col2, col3 = st.columns(3)
        #
        # with col1:
        #     st.metric(
        #         "PDFs",
        #         uploaded_file.name
        #     )
        #
        # with col2:
        #     st.metric(
        #         "Pages",
        #         total_pages
        #     )
        #
        # with col3:
        #     st.metric(
        #         "Chunks",
        #         len(chunks)
        #     )


    #chat history


    for message in st.session_state.messages:
        with st.chat_message(
                message["role"]
        ):
            st.write(
                message["content"]
            )


    # QUESTION SECTION


    st.subheader("Ask Questions")

    question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)


        # Question Embedding
        question_embedding = embedding_model.encode(
            question
        )

        question_embedding = np.array(
            [question_embedding]
        ).astype("float32")

        #FIASS SEARCH

        k = min(
            3,
            len(chunks)
        )

        distances, indices = index.search(
            question_embedding,
            k
        )
        # st.write("Indices:", indices)
        # st.write("Distances:", distances)


        # Retrieve Chunks
        retrieved_chunks = []

        for idx in indices[0]:
            retrieved_chunks.append(
                chunks[idx]
            )

        # Combine Context
        context = "\n\n".join(
            retrieved_chunks
        )

        # Debug View
        SHOW_DEBUG = False
        if SHOW_DEBUG:
            with st.expander("Retrieved Context"):
                st.write(context)


        # Gemini Prompt
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context,
say:

"I could not find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

        # Gemini Response
        with st.spinner(
            "Generating answer..."
        ):

            response = gemini_model.generate_content(
                prompt
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.text
                }
            )
            with st.chat_message("assistant"):
                st.write(response.text)


        # Final Answer
        st.subheader("Answer")
        st.write(
            response.text
        )