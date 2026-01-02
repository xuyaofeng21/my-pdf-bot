import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="多文件 AI 助手", layout="wide")
st.title("📚 多文档 AI 智能问答助手")

# --- 2. 侧边栏：安全 Key + 多文件上传 ---
with st.sidebar:
    st.header("⚙️ 设置面板")

    # === 🔒 安全改进部分 Start ===
    # 逻辑：优先读 Secrets，不把 Key 显示在输入框里
    api_key = None

    if "DEEPSEEK_API_KEY" in st.secrets:
        # 如果云端有 Key，直接用，不回显
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        st.success("✅ 云端密钥已激活")
        st.info("系统已自动加载密钥，无需手动输入。")
    else:
        # 如果没有，才显示输入框
        api_key = st.text_input("请输入 DeepSeek API Key", type="password")
        if not api_key:
            st.warning("⚠️ 请输入密钥以开始使用")
    # === 🔒 安全改进部分 End ===

    st.markdown("---")

    # === 📂 多文件改进部分 Start ===
    # accept_multiple_files=True 允许选多个
    uploaded_files = st.file_uploader(
        "上传 PDF 文件 (支持多个)",
        type=["pdf"],
        accept_multiple_files=True
    )
    # === 📂 多文件改进部分 End ===

    process_button = st.button("🚀 开始分析文档")


# --- 3. 核心函数：处理多个 PDF ---
def get_pdf_text(pdf_docs):
    text = ""
    # 循环遍历每一个上传的文件
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # 容错处理：有些页可能是空的
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # 使用本地轻量级模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# --- 4. 主逻辑 ---
if process_button and uploaded_files and api_key:
    with st.spinner("正在疯狂阅读所有文档..."):
        # 1. 提取所有 PDF 的文字
        raw_text = get_pdf_text(uploaded_files)

        # 2. 切片
        text_chunks = get_text_chunks(raw_text)

        # 3. 存入数据库
        # 注意：这里我们用 st.session_state 把数据库存起来，防止每次提问都重新算
        vector_store = get_vector_store(text_chunks)
        st.session_state.vector_store = vector_store

        st.success(f"✅ 处理完成！共读取了 {len(uploaded_files)} 个文件。")

# --- 5. 聊天界面 ---
if "vector_store" in st.session_state:
    st.markdown("### 💬 开始提问")
    user_question = st.text_input("关于这些文档，你想问什么？")

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

        # (可选) 显示参考了哪一段
        with st.expander("查看参考来源"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
else:
    if not uploaded_files:
        st.info("👈 请先在左侧上传 PDF 文件")