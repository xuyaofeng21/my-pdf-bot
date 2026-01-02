import streamlit as st
from pypdf import PdfReader

# === 🛡️ 稳定版(0.1.x) 经典引用写法 ===
# 这些路径在 LangChain 0.1.20 版本里是绝对存在的
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# ==========================================

import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="多文件 AI 助手", layout="wide")
st.title("📚 多文档 AI 智能问答助手")

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 设置")

    api_key = None
    if "DEEPSEEK_API_KEY" in st.secrets:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        st.success("✅ 云端密钥已激活")
    else:
        api_key = st.text_input("DeepSeek API Key", type="password")

    st.markdown("---")
    uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("🚀 开始分析")


# --- 3. 核心功能 ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t: text += t
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # 使用本地模型，规避 OpenAiEmbeddings 收费问题
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# --- 4. 执行逻辑 ---
if process_button and uploaded_files and api_key:
    with st.spinner("正在处理文档..."):
        raw_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.vector_store = vector_store
        st.success("✅ 处理完成！")

# --- 5. 问答逻辑 ---
if "vector_store" in st.session_state:
    st.markdown("### 💬 提问")
    user_question = st.text_input("你想问什么？")

    if user_question:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": user_question})

        st.write("🤖 **AI 回答:**")
        st.write(response["result"])

        with st.expander("查看来源"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
else:
    if not uploaded_files:
        st.info("👈 请先在左侧上传 PDF 文件")