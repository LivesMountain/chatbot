# 企业知识库与问答机器人（LangChain + Python）

这是一个面向企业内部场景的 **RAG（检索增强生成）问答系统**，用于基于：
- 产品手册
- 技术文档
- 制度规范

进行精准问答，帮助员工或用户快速获取标准答案，降低咨询成本。

## 1. 主要能力

- 自动读取 `knowledge_base/` 目录下的 `txt/md/pdf` 文档。
- 使用 `OpenAIEmbeddings + Chroma` 建立本地向量索引。
- 使用 `ChatOpenAI` + 检索器（Retriever）实现基于企业知识的问答。
- 支持两种方式调用：
  - CLI 命令行对话
  - FastAPI HTTP 服务

## 2. 目录建议

```bash
.
├── app.py
├── requirements.txt
└── knowledge_base/
    ├── 产品手册.md
    ├── 技术规范.txt
    └── 制度文档.pdf
```

## 3. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 配置环境变量

创建 `.env` 文件（至少需要 OpenAI Key）：

```env
OPENAI_API_KEY=your_api_key
KB_PATH=./knowledge_base
CHROMA_DIR=./chroma_store
CHAT_MODEL=gpt-4o-mini
```

## 5. 启动方式

### 5.1 CLI 模式

```bash
python app.py --mode cli
```

### 5.2 API 模式

```bash
python app.py --mode api --host 0.0.0.0 --port 8000
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"报销审批流程是什么？"}'
```

## 6. 生产建议

- 文档更新后可定期重建向量索引。
- 对敏感信息增加权限控制（按部门/角色检索）。
- 将回答与来源文档一起展示，增强可信度与可审计性。
