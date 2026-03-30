"""
企业知识库 + 问答机器人（RAG）示例。

功能说明：
1. 读取内部产品手册、技术文档、制度规范。
2. 将文档切分并写入向量数据库（Chroma）。
3. 基于 LangChain 构建检索增强生成（RAG）问答链。
4. 提供 CLI 与 HTTP API 两种使用方式，便于集成到内部系统。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 加载环境变量（如 OPENAI_API_KEY）
load_dotenv()


class EnterpriseRAGBot:
    """企业知识库问答机器人。

    该类负责：
    - 文档加载
    - 文本切分
    - 向量索引构建
    - 问答链创建与调用
    """

    def __init__(
        self,
        docs_path: str = "./knowledge_base",
        persist_dir: str = "./chroma_store",
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.docs_path = Path(docs_path)
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.vector_store: Chroma | None = None
        self.prompt_template: PromptTemplate | None = None
        self.retriever = None
        self.llm: ChatOpenAI | None = None

    def _load_documents(self) -> list[Any]:
        """加载知识库目录内的文档。

        支持：
        - .txt / .md
        - .pdf
        """
        if not self.docs_path.exists():
            raise FileNotFoundError(f"知识库目录不存在: {self.docs_path}")

        # 加载纯文本与 Markdown
        text_loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.[tm][dx][td]",  # 匹配 .txt/.md
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        text_docs = text_loader.load()

        # 加载 PDF 文档
        pdf_loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        pdf_docs = pdf_loader.load()

        docs = text_docs + pdf_docs
        if not docs:
            raise ValueError(
                "未检测到可用文档。请在 knowledge_base 目录放入 txt/md/pdf 文档。"
            )
        return docs

    def _build_vector_store(self) -> None:
        """将文档向量化并写入 Chroma。"""
        docs = self._load_documents()

        # 文本切分：控制每段长度和重叠，提升检索召回效果
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
        )
        self.vector_store.persist()

    def _load_or_create_vector_store(self) -> None:
        """优先加载本地索引，不存在则自动创建。"""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        if Path(self.persist_dir).exists() and any(Path(self.persist_dir).iterdir()):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
        else:
            self._build_vector_store()

    def build_qa_chain(self) -> None:
        """构建 RAG 所需组件（检索器 + 提示词 + 大模型）。"""
        self._load_or_create_vector_store()
        assert self.vector_store is not None

        # 提示词强调仅基于内部资料回答，降低幻觉风险
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "你是企业内部知识库助手，需基于提供的上下文回答问题。\n"
                "要求：\n"
                "1) 优先引用产品手册、技术文档、制度规范。\n"
                "2) 若上下文不足，明确回答‘我暂时无法从知识库中确认该信息’。\n"
                "3) 回答尽量结构化，便于员工快速执行。\n\n"
                "上下文：\n{context}\n\n"
                "问题：{question}\n\n"
                "请给出准确、简洁、可执行的答案。"
            ),
        )

        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

    def ask(self, question: str) -> dict[str, Any]:
        """执行问答并返回答案 + 来源文档。"""
        if self.prompt_template is None or self.retriever is None or self.llm is None:
            self.build_qa_chain()
        assert self.prompt_template is not None
        assert self.retriever is not None
        assert self.llm is not None

        source_documents = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_documents)
        prompt_text = self.prompt_template.format(context=context, question=question)
        llm_response = self.llm.invoke(prompt_text)

        answer = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )
        sources = [doc.metadata.get("source", "未知来源") for doc in source_documents]
        return {
            "answer": answer,
            "sources": sorted(set(sources)),
        }


# ------------------------ FastAPI 服务封装 ------------------------
app = FastAPI(title="企业知识库问答机器人", version="1.0.0")
rag_bot = EnterpriseRAGBot()


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_api(payload: AskRequest) -> dict[str, Any]:
    """HTTP 接口：接收问题并返回答案。"""
    return rag_bot.ask(payload.question)


def run_cli() -> None:
    """命令行交互模式。"""
    bot = EnterpriseRAGBot(
        docs_path=os.getenv("KB_PATH", "./knowledge_base"),
        persist_dir=os.getenv("CHROMA_DIR", "./chroma_store"),
        model_name=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    )

    print("企业知识库问答机器人已启动，输入 q 退出。")
    while True:
        q = input("\n请输入问题 > ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break
        if not q:
            continue

        response = bot.ask(q)
        print("\n答案：")
        print(response["answer"])
        print("\n参考来源：")
        for src in response["sources"]:
            print(f"- {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description="企业知识库 RAG 问答系统")
    parser.add_argument(
        "--mode",
        choices=["cli", "api"],
        default="cli",
        help="运行模式：cli（命令行）或 api（HTTP 服务）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        import uvicorn

        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
