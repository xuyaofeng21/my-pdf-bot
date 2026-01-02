import streamlit as st
from pypdf import PdfReader

# === 🛡️ 稳定版(0.1.x) 引用保持不变 ===
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="多文件 AI 助手", layout="wide")
st.title("📚 多文档 AI 智能问答助手")

# --- 2. 侧边栏：设置与上传 ---
with st.sidebar:
    st.header("⚙️ 设置面板")

    # 获取密钥
    api_key = None
    if "DEEPSEEK_API_KEY" in st.secrets:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        st.success("✅ 云端密钥已激活")
    else:
        api_key = st.text_input("DeepSeek API Key", type="password")

    st.markdown("---")
    uploaded_files = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=True)

    # 处理按钮
    process_button = st.button("🚀 开始建库 (上传后点我)")

    st.markdown("---")
    # 添加一个清空历史的按钮
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()


# --- 3. 核心函数 (逻辑不变) ---
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# --- 4. 业务逻辑：处理文件 ---
if process_button and uploaded_files and api_key:
    with st.spinner("正在疯狂阅读文档，请稍候..."):
        raw_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        # 存入 Session
        st.session_state.vector_store = vector_store
        st.success("✅ 文档已处理完毕！现在可以在右侧提问了。")

# --- 5. 业务逻辑：聊天界面 (重点修改部分) ---

# 初始化聊天历史 (如果还没有的话)
if "messages" not in st.session_state:
    st.session_state.messages = []

# A. 把历史消息画在屏幕上
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# B. 等待用户输入 (这是新的输入框组件)
if prompt := st.chat_input("请根据文档提问..."):
    # 1. 还没传文件就想提问？拦截！
    if "vector_store" not in st.session_state:
        st.error("请先在左侧上传 PDF 并点击“开始建库”！")
        st.stop()

    # 2. 显示用户的话
    st.chat_message("user").markdown(prompt)
    # 记入小本本
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. AI 思考并回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 准备 LLM
            llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model_name="deepseek-chat",
                temperature=0.3
            )
            # 准备问答链
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(),
                return_source_documents=True
            )

            # 获取答案
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            # 显示答案
            st.markdown(result)

            # (可选) 显示来源，折叠起来不占地方
            with st.expander("查看参考来源"):
                for doc in response["source_documents"]:
                    st.write(doc.page_content)

            # 记入小本本
            st.session_state.messages.append({"role": "assistant", "content": result})