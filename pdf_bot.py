import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="PDF 智能问答", layout="wide")
st.title("📄 PDF 智能问答助手")

# --- 2. 侧边栏：上传文件 & 设置 ---

with st.sidebar:
    st.header("1. 上传文件")

    # 1. 尝试从云端 Secrets 里拿 Key
    # 注意：这里的名字 "DEEPSEEK_API_KEY" 必须和你 Secrets 里填的一模一样
    if "DEEPSEEK_API_KEY" in st.secrets:
        default_key = st.secrets["DEEPSEEK_API_KEY"]
        key_source = "✅ 已自动加载云端密钥"
    else:
        default_key = ""
        key_source = "⚠️ 未检测到云端密钥"

    # 2. 显示状态提示
    st.caption(key_source)

    # 3. 创建输入框
    # 如果找到了 Secret，value 就是那个 Key，用户就不用填了
    # 如果没找到，value 为空，用户需要手动填
    api_key = st.text_input("DeepSeek API Key", value=default_key, type="password")

    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

    st.markdown("---")
    st.markdown("### 🛠️ 处理状态")
    status_text = st.empty()

# --- 3. 核心逻辑：处理 PDF (如果用户上传了新文件) ---
# 定义一个路径来存数据库，跟之前的区分开
DB_PATH = "../pdf_chroma_db"


def process_pdf(uploaded_file):
    """读取PDF -> 切分 -> 存入向量库"""
    # a. 先把上传的文件存成临时文件
    temp_file_path = "../temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # b. 加载 PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # c. 切分文档 (Recursive 是更高级的切分器，不仅看字数，还看句号段落)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # d. 向量化并入库
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 创建一个新的数据库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    return vectorstore


# --- 4. 初始化 Session State (记忆) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- 5. 只有当用户点击上传，且数据库没准备好时，才去处理 ---
if uploaded_file and st.session_state.vector_db is None:
    if not api_key:
        st.error("请先输入 API Key！")
    else:
        with st.spinner("正在阅读 PDF，请稍等... (第一次可能会下载模型)"):
            try:
                # 调用上面的函数
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("PDF 处理完成！现在可以提问了。")
            except Exception as e:
                st.error(f"处理失败: {e}")

# --- 6. 聊天界面 ---
# 显示历史记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 处理用户提问
user_input = st.chat_input("在这个 PDF 里找什么？")

if user_input:
    # 检查有没有 Key 和 数据库
    if not api_key:
        st.warning("请先设置 API Key")
        st.stop()
    if st.session_state.vector_db is None:
        st.warning("请先上传 PDF 文件")
        st.stop()

    # A. 显示用户问题
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # B. 核心 RAG 检索流程
    with st.chat_message("assistant"):
        with st.spinner("AI 正在翻书查找..."):
            # 1. 在数据库里搜
            db = st.session_state.vector_db
            docs = db.similarity_search(user_input, k=2)  # 找最相似的2个片段

            if not docs:
                context = "没有在文档中找到相关信息。"
            else:
                # 把找到的文字拼起来
                context = "\n\n".join([d.page_content for d in docs])

            # 2. 组装 Prompt
            prompt = f"""
            你是一个文档助手。基于以下【参考资料】回答用户问题。

            【参考资料】：
            {context}

            【用户问题】：
            {user_input}
            """

            # 3. 调用 DeepSeek
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个乐于助人的助手。"},
                    {"role": "user", "content": prompt}
                ]
            )

            # 4. 显示答案
            answer = response.choices[0].message.content
            st.write(answer)

            # 5. 既然是 RAG，最好展示一下参考了哪一段（显得专业）
            with st.expander("查看 AI 参考的原文片段"):
                st.write(context)

            st.session_state.messages.append({"role": "assistant", "content": answer})