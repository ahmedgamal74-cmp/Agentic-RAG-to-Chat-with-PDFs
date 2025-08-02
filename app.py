# import streamlit as st
# import os

# from rag_engine import parse_pdfs


# st.title("Agentic RAG Chat")

# # Create docs directory if not exists
# os.makedirs("docs", exist_ok=True)

# uploaded_files = st.file_uploader(
#     "Upload PDF files", type="pdf", accept_multiple_files=True
# )

# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         # Save each uploaded file to docs/ directory
#         with open(f"docs/{uploaded_file.name}", "wb") as f:
#             f.write(uploaded_file.getbuffer())
#     st.success("File(s) uploaded successfully.")

#     # Optionally show extracted text for debugging
#     parsed_pages = parse_pdfs("docs")
#     st.write(f"Extracted {len(parsed_pages)} pages.")
#     if st.checkbox("Show first page text preview"):
#         st.write(parsed_pages[0] if parsed_pages else "No text extracted.")




# # Optional: Add a checkbox for user to preview the first page’s text
#     if parsed_pages:
#         if st.checkbox("Show preview of the first parsed page"):
#             st.write(f"File: {parsed_pages[0]['file_name']} | Page: {parsed_pages[0]['page_num']}")
#             st.write(parsed_pages[0]['text'] or "No text extracted from this page.")

#         # Optional: Let user select which page to preview
#         page_options = [f"{p['file_name']} - Page {p['page_num']}" for p in parsed_pages]
#         selected_idx = st.selectbox("Preview a parsed page:", range(len(page_options)), format_func=lambda x: page_options[x])
#         st.write(parsed_pages[selected_idx]['text'] or "No text extracted from this page.")
#     else:
#         st.warning("No text extracted from uploaded PDFs.")




import streamlit as st
import os
from rag_engine import parse_pdfs, build_index, load_index
from dotenv import load_dotenv
from llama_index.llms.huggingface import HuggingFaceLLM
import json
load_dotenv()


def get_local_llm():
    return HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        device_map="auto",
    )

st.title("Agentic RAG Chat")

os.makedirs("docs", exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(f"docs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("File(s) uploaded successfully.")


    parsed_chunks = parse_pdfs("docs")
    st.write(f"Extracted {len(parsed_chunks)} chunks from uploaded PDFs.")


    with open("parsed_chunks.json", "w", encoding="utf-8") as f:
        json.dump(parsed_chunks, f, ensure_ascii=False, indent=2)


    # Add "Build Index" button
    if st.button("Build Index"):
        with st.spinner("Building semantic index..."):
            build_index(parsed_chunks, persist_dir="storage")
        st.success("Index built and stored successfully.")





if os.path.exists("storage"):
    st.header("Ask Questions About Your PDFs")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your question and press Enter:")

    if user_input:
        with st.spinner("Searching and generating answer..."):
            index = load_index("storage")
            llm = get_local_llm()
            try:
                query_engine = index.as_query_engine(llm=llm, similarity_top_k=1)
                result = query_engine.query(user_input)
            except Exception as e:
                st.error(f"Error running local LLM: {e}")
                result = None

        answer = str(result) if result else "No answer found."
        sources = []
        if result and hasattr(result, "source_nodes") and result.source_nodes:
            for node in result.source_nodes:
                meta = node.node.metadata
                file = meta.get("file_name")
                page = meta.get("page_num")
                chunk_num = meta.get("chunk_num")
                # More precise: file, page, chunk
                if file and page and chunk_num:
                    sources.append(f"{file} (Page {page}, Chunk {chunk_num})")
                elif file and page:
                    sources.append(f"{file} (Page {page})")
        if not sources or not answer.strip():
            answer = "No answer found."
            sources = []

        st.session_state.chat_history.append(
            {"question": user_input, "answer": answer, "sources": sources}
        )

    if st.session_state.chat_history:
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Agent:** {chat['answer']}")
            if chat["sources"]:
                st.markdown("Citations: " + ", ".join(chat["sources"]))
            st.markdown("---")