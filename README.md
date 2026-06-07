# localrag — 离线 AI 聊天助手

> 零安装、解压即用、CPU 就能跑的本地 AI 聊天助手

## 特点

- 🔌 **解压即用** — 内置便携 Python，不写注册表，不污染系统
- 🧠 **本地推理** — Qwen2.5-1.5B 量化模型，纯 CPU，无需 GPU
- 📚 **RAG 知识库** — 拖拽上传文档，自动分块向量化，对话时语义检索
- 🎭 **5 种人格** — 小暖/冷冷/老铁/阿屿/张雪峰，含张雪峰私有知识库
- 🎨 **5 套主题** — 暗夜紫/薄荷绿/暖橙/深海蓝/极简白
- 💾 **对话持久化** — JSON 自动保存，下拉加载历史
- ⚙️ **参数可调** — 温度/Top-P/上下文窗口全可拖滑块

## 快速开始（使用者 — 解压即用）

1. 下载完整包（约 3.2GB）
   - **百度网盘：** [链接待上传]
   - 或从 [GitHub Releases](../../releases) 下载
2. 解压到**纯英文路径**（路径不要有中文）
3. 双击 `启动助手.bat`
4. 等 20 秒左右，浏览器自动打开 `http://localhost:7860`
5. 开始聊天

**不需要装 Python、pip、Ollama、Docker 或任何东西。**

## 快速开始（开发者 — 从源码跑）

```bash
# 1. 克隆源码
git clone https://github.com/chenbeixuan793/localrag.git
cd localrag

# 2. 准备 Python 环境（需 Python 3.10+）
pip install -r requirements.txt

# 3. 下载模型和推理引擎
python -m localrag.download --all

# 4. 启动
python run.py
```

> 如果网络受限无法下载，参见下方"离线安装"说明。

```bash
git clone https://github.com/chenbeixuan793/localrag.git
cd localrag
pip install -r requirements.txt

# 下载模型
python -m localrag.download --chat-model --embed-model --llama-server

# 启动
python run.py
```

## 架构

```
用户输入 → embedder.embed_query()
         → vectordb.search_text()        ← 用户知识库
         → vectordb.search_persona()     ← 人格私有知识库
         → engine.chat()                 ← llama-server /v1/chat/completions
         → 流式输出 → Chatbot
```

```
localrag/
├── app.py          ← Gradio 6.x Web UI (1550+ 行)
├── engine.py       ← 推理引擎 (llama-server HTTP API)
├── personality.py  ← 五层人格系统 + 5种预设
├── vectordb.py     ← ChromaDB 封装 (用户KB + 人格私有KB)
├── embedder.py     ← 嵌入模型 (bge-small-zh-v1.5, 512维)
├── ingest.py       ← 文档解析 + 智能分块
├── config.py       ← 全配置 dataclass
├── download.py     ← 模型下载 (hf-mirror + gh-proxy)
└── cli.py          ← 命令行工具
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 聊天模型 | Qwen2.5-1.5B-Instruct GGUF Q4_K_M |
| 推理引擎 | llama-server (llama.cpp) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | ChromaDB (SQLite) |
| Web UI | Gradio 6.x |
| 文档解析 | python-docx + PyPDF2 + pdfplumber |

## 配置

```python
# config.py
chat_n_ctx: 16384       # 上下文窗口
chat_n_threads: 8       # CPU 推理线程
chat_n_batch: 1024      # 批处理大小
temperature: 0.7        # 生成随机性
max_tokens: 2048        # 单次最大回复
rag_top_k: 5            # RAG 检索条数
```

## 人格系统

基于五层结构（硬规则 → 表达风格 → 情感模式 → 性格特质 → 关系记忆）。

| 人格 | 描述 |
|------|------|
| 💛 小暖 | 温暖知心朋友（默认） |
| 💜 冷冷 | 傲娇毒舌选手 |
| 🤜 老铁 | 损友铁哥们 |
| 🌙 阿屿 | 灵魂伴侣（深夜模式） |
| 🎙️ 张雪峰 | 升学就业顾问（含 33K 字私有知识库） |

## License

MIT

## 免责声明

本项目中的人格模拟（包括张雪峰角色）基于公开言论推断，仅供学习和个人使用。不代表本人或其家属观点。请勿用于商业用途或冒充真人。
