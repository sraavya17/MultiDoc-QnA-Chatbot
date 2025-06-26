import os
import streamlit as st
import tempfile
from main import load_documents, process_documents, model_init, create_qa_chain

# Set the session state for the qa_chain and vector_store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.create_qa_chain = None

# Streamlit app configuration
st.set_page_config(page_title="Document Q&A", page_icon=":robot;", layout="wide")
st.title("Multi-Document Q&A")
st.write("Upload your documents and ask questions based on their content.")
st.write("Supported formats: PDFs, Text files")

# File uploader for multiple documents
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        with st.spinner("processing documents..."):
            try:
                # Save uploaded files to a temporary directory
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                # Load and process documents
                documents = load_documents(file_paths)
                st.session_state.vector_store = process_documents(documents)
                llm = model_init()
                st.session_state.qa_chain = create_qa_chain(llm, st.session_state.vector_store)
                st.success(f"Processed {len(file_paths)} documents successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
    else:
        st.warning("Please upload at least one document to process.")

# Question input and answer display
if st.session_state.get('qa_chain'):
    st.divider()
    st.header("Ask a Question")
    question = st.text_input("Enter your question here:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                result = st.session_state.qa_chain({"query": question})
                st.subheader("Answer:")
                st.write(result['result'])
                with st.expander("Source Documents"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.caption(f"Document{i+1}: {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content[:500]+"...")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.warning("Please process documents first to ask questions.")


