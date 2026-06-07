"""Gradio Web UI — 好看的聊天界面 + 知识库管理 + 参数调节"""

import os
import time
from pathlib import Path
from typing import Optional, Generator

import gradio as gr

from .config import get_config, reset_config, AppConfig
from .engine import ChatEngine, get_engine
from .embedder import Embedder, get_embedder
from .vectordb import VectorDB, get_vectordb
from .ingest import ingest_file, ingest_text
from .personality import PRESET_PERSONAS, build_system_prompt, get_preset

# ============================================================
# 主题系统 — 4套配色，CSS 变量驱动
# ============================================================

THEMES = {
    "purple": {
        "name": "🌙 暗夜紫",
        "vars": {
            "--bg-primary": "#0f0f19", "--bg-secondary": "#1a1a2e",
            "--bg-card": "#252540", "--bg-input": "#2d2d4a",
            "--text-primary": "#f0f0ff", "--text-secondary": "#b0b0d0",
            "--accent": "#a78bfa", "--accent-hover": "#8b5cf6", "--accent-light": "#c4b5fd",
            "--border": "#3a3a55", "--success": "#34d399", "--warning": "#fbbf24", "--danger": "#f87171",
            "--radius": "12px", "--glow": "rgba(139,92,246,0.3)",
        },
        "gradient": "linear-gradient(135deg, #c4b5fd, #a78bfa, #f472b6)",
    },
    "mint": {
        "name": "🍃 薄荷绿",
        "vars": {
            "--bg-primary": "#0d1a15", "--bg-secondary": "#152e24",
            "--bg-card": "#1e3d30", "--bg-input": "#264a3a",
            "--text-primary": "#e8faf0", "--text-secondary": "#90c9a8",
            "--accent": "#4ade80", "--accent-hover": "#22c55e", "--accent-light": "#86efac",
            "--border": "#2a4a35", "--success": "#34d399", "--warning": "#fbbf24", "--danger": "#f87171",
            "--radius": "14px", "--glow": "rgba(74,222,128,0.3)",
        },
        "gradient": "linear-gradient(135deg, #86efac, #4ade80, #22c55e)",
    },
    "warm": {
        "name": "☀️ 暖橙",
        "vars": {
            "--bg-primary": "#1c140a", "--bg-secondary": "#2a1f10",
            "--bg-card": "#382a18", "--bg-input": "#453420",
            "--text-primary": "#fff8ee", "--text-secondary": "#c9a87c",
            "--accent": "#fbbf24", "--accent-hover": "#f59e0b", "--accent-light": "#fcd34d",
            "--border": "#4a3820", "--success": "#34d399", "--warning": "#fbbf24", "--danger": "#f87171",
            "--radius": "10px", "--glow": "rgba(251,191,36,0.35)",
        },
        "gradient": "linear-gradient(135deg, #fcd34d, #fbbf24, #f97316)",
    },
    "ocean": {
        "name": "🌊 深蓝",
        "vars": {
            "--bg-primary": "#0b1120", "--bg-secondary": "#151d35",
            "--bg-card": "#1e2a4a", "--bg-input": "#263558",
            "--text-primary": "#e8eeff", "--text-secondary": "#90a0cc",
            "--accent": "#60a5fa", "--accent-hover": "#3b82f6", "--accent-light": "#93bbfc",
            "--border": "#2a3a58", "--success": "#34d399", "--warning": "#fbbf24", "--danger": "#f87171",
            "--radius": "8px", "--glow": "rgba(96,165,250,0.35)",
        },
        "gradient": "linear-gradient(135deg, #93bbfc, #60a5fa, #a78bfa)",
    },
    "plain": {
        "name": "⬜ 极简",
        "vars": {
            "--bg-primary": "#1e1e1e", "--bg-secondary": "#2a2a2a",
            "--bg-card": "#353535", "--bg-input": "#3d3d3d",
            "--text-primary": "#f5f5f5", "--text-secondary": "#aaaaaa",
            "--accent": "#888888", "--accent-hover": "#777777", "--accent-light": "#bbbbbb",
            "--border": "#444444", "--success": "#5a5a5a", "--warning": "#888888", "--danger": "#777777",
            "--radius": "4px", "--glow": "rgba(150,150,150,0.25)",
        },
        "gradient": "linear-gradient(135deg, #bbb, #999, #777)",
    },
}


def build_theme_css(theme_key: str) -> str:
    """构建主题 CSS + JS（JS 直接改 CSS 变量，绕过 Gradio 优先级问题）"""
    t = THEMES.get(theme_key, THEMES["purple"])
    v = t["vars"]
    g = t["gradient"]

    return f"""<style>
:root {{
{chr(10).join(f'  {k}: {v} !important;' for k, v in v.items())}
}}
.theme-injector {{ display: none !important; }}
body {{ background: var(--bg-primary) !important; margin: 0 !important; padding: 0 !important; }}
.gradio-container, .gradio-container .main, .gradio-container .contain, .app {{ max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 8px 16px !important; background: var(--bg-primary) !important; }}
.gradio-container > .contain {{ padding: 0 !important; }}
/* 覆盖 Gradio 组件底色 */
.gr-box, .gr-group, .gr-panel, .panel, [class*=\"gr-group\"], [class*=\"gr-box\"] {{ background: var(--bg-secondary) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }}
.gr-prose, .prose, .markdown, [class*=\"prose\"] {{ color: var(--text-primary) !important; }}
.chatbot {{ border-radius: var(--radius) !important; border: 1px solid var(--border) !important; background: var(--bg-secondary) !important; }}
.chatbot .message {{ border-radius: var(--radius) !important; padding: 12px 16px !important; margin: 8px 0 !important; }}
.chatbot .user {{ background: var(--accent) !important; color: #fff !important; }}
.chatbot .bot, .bubble.bot, .message.bot, [class*=\"bubble\"] [class*=\"bot\"] {{ color: #1a1a1a !important; }}
.chatbot .bot, .bubble.bot {{ background: #f0f0f5 !important; border: 1px solid #d0d0d8 !important; }}
label, .label, .gr-label {{ color: var(--text-secondary) !important; }}
.gr-dropdown, .gr-slider, select {{ background: var(--bg-input) !important; color: var(--text-primary) !important; }}
.app-header {{ text-align: center; padding: 20px 0 10px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }}
.app-header h1 {{ font-size: 2em; font-weight: 700; background: {g}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }}
.app-header p {{ color: var(--text-secondary); font-size: 0.9em; margin: 4px 0 0; }}
.chatbot {{ border-radius: var(--radius) !important; border: 1px solid var(--border) !important; background: var(--bg-secondary) !important; }}
.chatbot .message {{ border-radius: var(--radius) !important; padding: 12px 16px !important; margin: 8px 0 !important; }}
.chatbot .user {{ background: var(--accent) !important; color: #fff !important; }}
.chatbot .bot, .bubble.bot, .message.bot, [class*=\"bubble\"] [class*=\"bot\"] {{ color: #1a1a1a !important; }}
.chatbot .bot, .bubble.bot {{ background: #f0f0f5 !important; border: 1px solid #d0d0d8 !important; }}
textarea, input[type="text"] {{ border-radius: var(--radius) !important; border: 1px solid var(--border) !important; background: var(--bg-input) !important; color: var(--text-primary) !important; padding: 12px 16px !important; font-size: 1em !important; }}
textarea:focus, input[type="text"]:focus {{ border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--glow) !important; }}
button {{ border-radius: var(--radius) !important; font-weight: 600 !important; transition: all 0.2s !important; }}
button.primary {{ background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important; border: none !important; color: #fff !important; box-shadow: 0 4px 14px var(--glow) !important; }}
button.primary:hover {{ transform: translateY(-1px); box-shadow: 0 6px 20px var(--glow) !important; }}
button.secondary {{ background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; }}
button.secondary:hover {{ border-color: var(--accent) !important; }}
input[type="range"] {{ accent-color: var(--accent); }}
.footer {{ text-align: center; padding: 20px 0; color: var(--text-secondary); font-size: 0.8em; border-top: 1px solid var(--border); margin-top: 20px; }}
.footer .status {{ color: var(--success); }}
.footer .dot {{ display: inline-block; width: 8px; height: 8px; background: var(--success); border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
</style>"""


# 默认主题 CSS（初始加载用）
DEFAULT_THEME_CSS = build_theme_css("purple")

# ============================================================
# 全局状态
# ============================================================

class AppState:
    """应用全局状态"""
    def __init__(self):
        self.engine: Optional[ChatEngine] = None
        self.embedder: Optional[Embedder] = None
        self.vectordb: Optional[VectorDB] = None
        self.current_persona: str = "warm-friend"
        self.knowledge_enabled: bool = True
        self.is_ready: bool = False

    def init_components(self, progress=None) -> str:
        """初始化所有组件（可带 progress 对象或返回纯文本）"""
        logs = []
        try:
            if progress:
                progress(0.15, desc="加载嵌入模型...")
            self.embedder = get_embedder()
            self.embedder.load()
            logs.append("✅ 嵌入模型就绪 (bge-small-zh-v1.5)")

            if progress:
                progress(0.35, desc="初始化知识库...")
            self.vectordb = get_vectordb()
            self.vectordb.load()
            logs.append(f"✅ 知识库就绪 ({self.vectordb.count} 个文档块)")

            if progress:
                progress(0.5, desc="加载聊天模型...")
            self.engine = get_engine()
            model_path = Path("models") / get_config().model.chat_model_file
            if model_path.exists():
                logs.append(f"🧠 正在加载聊天模型 (1.5B GGUF, CPU)...")
                if progress:
                    progress(0.6, desc="启动推理引擎...")
                self.engine.load(model_path)
                logs.append("✅ 聊天模型就绪")
            else:
                logs.append("⚠️ 聊天模型未找到，请先下载")

            if self.engine:
                if progress:
                    progress(0.9, desc="设置人格...")
                self.engine.set_personality(self.current_persona)

            if progress:
                progress(1.0, desc="就绪!")
            self.is_ready = True
            logs.append("\n🎉 一切就绪，开始聊天吧！")

        except Exception as e:
            logs.append(f"❌ 初始化失败: {e}")
            self.is_ready = False

        return "\n".join(logs)


_state = AppState()


# ============================================================
# 聊天逻辑
# ============================================================

# ============================================================
# 对话持久化 — JSON 文件存储，下拉列表加载
# ============================================================
import json
import datetime

CONV_DIR = Path("knowledge/conversations")
CONV_DIR.mkdir(parents=True, exist_ok=True)

_current_conv_file: Optional[str] = None  # 当前对话文件名


def _conv_path(filename: str) -> Path:
    return CONV_DIR / filename


def list_conversations() -> list[str]:
    """列出所有已保存的对话，按时间倒序"""
    files = sorted(CONV_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in files]


def _auto_save():
    """每次对话后自动保存"""
    global _current_conv_file
    if not _state.engine:
        return
    history = _state.engine.get_history()
    if not history:
        return
    cur = _current_conv_file
    if cur is None:
        first_msg = ""
        for h in history:
            if h["role"] == "user":
                first_msg = h["content"][:30].replace("/", "").replace("\\", "").strip()
                break
        if not first_msg:
            first_msg = "新对话"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _current_conv_file = f"{ts}_{first_msg}.json"
    save_path = _conv_path(_current_conv_file)
    save_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def new_conversation():
    """新建对话"""
    global _current_conv_file
    _current_conv_file = None
    if _state.engine:
        _state.engine.clear_history()
    return [], _update_conv_dropdown()


def save_conversation():
    """手动保存当前对话"""
    global _current_conv_file
    if not _state.engine:
        return _update_conv_dropdown()
    history = _state.engine.get_history()
    if not history:
        return _update_conv_dropdown()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    first_msg = ""
    for h in history:
        if h["role"] == "user":
            first_msg = h["content"][:30].replace("/", "").replace("\\", "").strip()
            break
    _current_conv_file = f"{ts}_{first_msg or 'saved'}.json"
    _conv_path(_current_conv_file).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return _update_conv_dropdown()


def load_conversation(filename: str):
    """加载对话"""
    global _current_conv_file
    if not filename:
        return [], _update_conv_dropdown()
    data = json.loads(_conv_path(filename).read_text(encoding="utf-8"))
    _current_conv_file = filename
    if _state.engine:
        _state.engine._conversation_history = data
    # 转为 Gradio 格式
    formatted = []
    for msg in data:
        formatted.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"],
        })
    return formatted, _update_conv_dropdown()


def delete_conversation(filename: str):
    """删除对话"""
    global _current_conv_file
    if not filename:
        return _update_conv_dropdown()
    p = _conv_path(filename)
    if p.exists():
        p.unlink()
    if _current_conv_file == filename:
        _current_conv_file = None
    return _update_conv_dropdown()


def _update_conv_dropdown() -> gr.Dropdown:
    """更新对话列表下拉框"""
    convs = list_conversations()
    choices = [(f.replace("_", " ").replace(".json", "")[:60], f) for f in convs]
    cur = _current_conv_file
    if cur and cur not in convs:
        choices.insert(0, (cur.replace(".json", ""), cur))
    return gr.Dropdown(choices=choices, value=cur, label="💬 历史对话")


def _format_history(history: list[dict]) -> list[dict]:
    """将内部历史转为 Gradio 格式"""
    formatted = []
    for msg in history:
        formatted.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"],
        })
    return formatted


def chat_fn(message: str, history: list, temperature: float, top_p: float,
            max_tokens: int, top_k: int, enable_knowledge: bool,
            context_window: int = 8192) -> Generator:
    """流式聊天核心函数"""
    if not _state.is_ready or not _state.engine:
        yield history + [{"role": "assistant", "content": "⚠️ 系统未就绪，请先初始化"}]
        return

    engine = _state.engine

    # 更新参数（含上下文窗口）
    engine.update_params(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                         top_k=top_k, context_window=context_window)

    # RAG 检索 — 用户知识库 + 人格私有知识库
    knowledge = None
    if enable_knowledge and _state.vectordb and _state.embedder:
        try:
            parts = []
            query_embedding = _state.embedder.embed_query(message)
            top_k = get_config().rag.top_k
            # 用户知识库
            if _state.vectordb.count > 0:
                user_results = _state.vectordb.search_text(query_embedding, top_k=top_k)
                if user_results:
                    parts.append("【你的知识库】\n" + "\n---\n".join(user_results))
            # 人格私有知识库
            persona_results = _state.vectordb.search_persona(
                _state.current_persona, query_embedding, top_k=top_k)
            if persona_results:
                parts.append("【人格参考知识】\n" + "\n---\n".join(persona_results))
            if parts:
                knowledge = "\n\n".join(parts)
        except Exception:
            pass

    # 流式生成
    full_response = ""
    history = history or []

    try:
        for token_chunk in _stream_chat(engine, message, knowledge):
            full_response += token_chunk
            yield history + [{"role": "assistant", "content": full_response}]
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"😢 出错了: {e}"}]


def _stream_chat(engine: ChatEngine, message: str, knowledge: Optional[str]) -> Generator[str, None, None]:
    """流式输出生成器（模拟流式）"""
    result = engine.chat(message=message, knowledge=knowledge, stream=False)

    # 模拟逐 token 输出（llama-cpp 的流式不直接支持 on_token）
    chunk_size = 3
    for i in range(0, len(result), chunk_size):
        yield result[i:i+chunk_size]
        time.sleep(0.02)


def clear_chat():
    """清空聊天"""
    if _state.engine:
        _state.engine.clear_history()
    return [], "✅ 对话已清空"


# ============================================================
# 知识库管理
# ============================================================

def upload_file(file, chunk_size: int, chunk_overlap: int):
    """上传文件到知识库"""
    if file is None:
        return "⚠️ 请选择文件", gr.update()

    try:
        path = Path(file.name)
        n = ingest_file(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        stats = _knowledge_stats()
        return f"✅ 已导入: {path.name} → {n} 个文档块", stats
    except Exception as e:
        return f"❌ 导入失败: {e}", gr.update()


def add_text_knowledge(text: str, source: str):
    """手动添加文本到知识库"""
    if not text.strip():
        return "⚠️ 请输入文本内容", gr.update()
    try:
        n = ingest_text(text, source)
        stats = _knowledge_stats()
        return f"✅ 已添加: {n} 个文档块", stats
    except Exception as e:
        return f"❌ 添加失败: {e}", gr.update()


def _knowledge_stats() -> str:
    """知识库概览——展示来源列表，仅拉元数据（不拉文档全文）"""
    if not _state.vectordb:
        return "📚 知识库未初始化"
    try:
        db = _state.vectordb
        count = db.count
        if count == 0:
            return "📚 知识库为空，上传文档开始吧"
        # 只拿 metadata，不拿 documents 字段，避免卡
        result = db.collection.get(include=["metadatas"], limit=500)
        sources = {}
        for m in (result.get("metadatas") or []):
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        lines = [f"📚 **{count}** 个文档块 | **{len(sources)}** 个来源\n"]
        lines.append("| 来源 | 块数 |")
        lines.append("|------|------|")
        for src, n in sorted(sources.items(), key=lambda x: -x[1])[:30]:
            lines.append(f"| {src[:50]} | {n} |")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 获取失败: {e}"


def _knowledge_detail() -> str:
    """知识库详细统计——仅在点击刷新按钮时调用"""
    if not _state.vectordb:
        return "📚 知识库未初始化"
    try:
        db = _state.vectordb
        docs = db.get_all_docs(limit=500)
        if not docs:
            return "📚 知识库为空"
        sources = {}
        for d in docs:
            src = d["meta"].get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        lines = [f"📚 共 **{len(docs)}** 个文档块\n"]
        lines.append("| 来源 | 块数 |")
        lines.append("|------|------|")
        for src, count in sorted(sources.items(), key=lambda x: -x[1])[:30]:
            lines.append(f"| {src[:50]} | {count} |")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 获取详情失败: {e}"


def clear_knowledge():
    """清空知识库"""
    if _state.vectordb:
        _state.vectordb.delete_all()
    return "🗑️ 知识库已清空", _knowledge_stats()


# ============================================================
# 设置与初始化
# ============================================================

def init_system():
    """自动初始化（页面加载时自动调用）"""
    if _state.is_ready:
        return "\n".join([
            "✅ 嵌入模型就绪 (bge-small-zh-v1.5)",
            f"✅ 知识库就绪 ({_state.vectordb.count if _state.vectordb else 0} 个文档块)",
            "✅ 聊天模型就绪",
            "\n🎉 一切就绪，开始聊天吧！",
        ])
    return _state.init_components(progress=None)


def _build_persona_kb(persona_key: str) -> str:
    """从参考文件构建人格私有知识库"""
    from .personality import get_preset
    preset = get_preset(persona_key)
    kb_dir = preset.get("knowledge_dir")
    if not kb_dir:
        return ""
    # 参考文件路径
    kb_path = Path(f"knowledge/persona/{kb_dir}")
    if not kb_path.exists():
        return "⚠️ 人格知识库文件未找到"
    try:
        from .ingest import _read_file, _split_text
        docs = []
        for f in sorted(kb_path.glob("*.md")):
            text, _ = _read_file(f)
            docs.extend(_split_text(text, chunk_size=800, chunk_overlap=150))
        if not docs:
            return ""
        embeddings = _state.embedder.embed(docs)
        n = _state.vectordb.create_persona_kb(persona_key, docs, embeddings)
        return f"✅ 已加载 {n} 条参考知识"
    except Exception as e:
        return f"⚠️ 知识库构建失败: {e}"


def switch_persona(persona_key: str):
    """切换人格（含私有知识库加载 + 清空对话历史）"""
    _state.current_persona = persona_key
    msg = ""
    if _state.engine:
        preset = _state.engine.set_personality(persona_key)
        _state.engine.clear_history()  # 清掉旧人格的对话，避免模型被历史带偏
        msg = f"✅ 已切换为: {preset['display']}"
    else:
        msg = "⚠️ 引擎未初始化"
    if _state.vectordb and _state.embedder:
        kb_msg = _build_persona_kb(persona_key)
        if kb_msg:
            msg += "\n" + kb_msg
    return msg, []  # 清空聊天界面


def get_persona_info(persona_key: str) -> str:
    """获取人格详情"""
    preset = get_preset(persona_key)
    name = preset["name"]
    desc = preset["description"]
    traits = preset.get("extra_traits", "")
    return f"### {name}\n{desc}\n\n{traits}" if traits else f"### {name}\n{desc}"


# ============================================================
# 构建 UI
# ============================================================

def create_ui():
    """创建 Gradio UI"""

    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="💫 离线 AI 聊天助手",
        fill_height=True,
    ) as demo:

        demo.theme = theme

        # ---- 头部 ----
        gr.HTML("""
        <div class="app-header">
            <h1>💫 离线 AI 聊天助手</h1>
            <p>完全本地运行 · 不吃配置 · 不调任何 API · 你的离线朋友</p>
        </div>
        """)

        # ---- 状态栏 ----
        with gr.Row(elem_classes="footer"):
            status_indicator = gr.Markdown(
                '<span class="dot"></span><span class="status">系统已就绪，开始聊天吧！</span>'
            )

        # ---- 主内容 ----
        with gr.Row(equal_height=False):
            # 左侧: 聊天区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_classes="chatbot",
                    height=550,
                    layout="bubble",
                    avatar_images=(None, None),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你想说的话... (Enter 发送, Shift+Enter 换行)",
                        scale=9,
                        lines=2,
                        label=None,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("发送 💬", variant="primary", scale=1, elem_classes="primary")

                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm", elem_classes="secondary")
                    knowledge_toggle = gr.Checkbox(value=True, label="📚 启用知识库检索", info="自动搜索你的文档")

            # 右侧: 控制面板
            with gr.Column(scale=1, min_width=320):
                # 主题 CSS 注入（用 elem_classes 隐藏容器但保留 style 生效）
                theme_css = gr.HTML(DEFAULT_THEME_CSS, elem_classes="theme-injector")

                with gr.Group():
                    gr.Markdown("### 🎨 界面主题")
                    theme_selector = gr.Dropdown(
                        choices=[(t["name"], k) for k, t in THEMES.items()],
                        value="purple",
                        label="选择主题",
                        interactive=True,
                    )

                with gr.Group():
                    gr.Markdown("### 🚀 系统控制")
                    init_btn = gr.Button("⚡ 初始化系统", variant="primary", size="lg", elem_classes="primary")
                    init_log = gr.Markdown("点击初始化按钮启动系统...")

                # 人格选择
                with gr.Group():
                    gr.Markdown("### 🎭 人格切换")
                    persona_choices = [(v["display"], k) for k, v in PRESET_PERSONAS.items()]
                    persona_dropdown = gr.Dropdown(
                        choices=persona_choices,
                        value="warm-friend",
                        label="选择助手人格",
                        interactive=True,
                    )
                    with gr.Accordion("📋 人格详情", open=False):
                        persona_info = gr.Markdown(get_persona_info("warm-friend"))
                    persona_btn = gr.Button("🔄 切换人格", variant="secondary", size="sm", elem_classes="secondary")
                    persona_msg = gr.Textbox(label="", visible=False)

                # 对话管理
                with gr.Group():
                    gr.Markdown("### 💬 对话管理")
                    conv_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="历史对话",
                        info="每次聊天自动保存",
                        interactive=True,
                    )
                    with gr.Row():
                        new_conv_btn = gr.Button("➕ 新建", variant="secondary", size="sm", elem_classes="secondary")
                        load_conv_btn = gr.Button("📂 加载", variant="secondary", size="sm", elem_classes="secondary")
                        del_conv_btn = gr.Button("🗑️ 删除", variant="secondary", size="sm", elem_classes="secondary")

                # 参数调节
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成参数")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature",
                                            info="越高越随机，越低越稳定")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P",
                                      info="核采样概率阈值")
                    top_k = gr.Slider(1, 100, value=40, step=1, label="Top-K",
                                      info="采样候选数")
                    max_tokens = gr.Slider(64, 4096, value=2048, step=64, label="Max Tokens",
                                           info="最大回复长度")
                    context_window = gr.Slider(2048, 16384, value=8192, step=2048,
                                               label="上下文窗口 (tokens)",
                                               info="越大记越多轮，越小响应越快")

        # ---- 知识库标签页 ----
        with gr.Column():
            with gr.Accordion("📁 上传文档"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="拖拽或点击上传文档",
                            file_types=[".pdf", ".docx", ".doc", ".md", ".txt",
                                       ".text", ".log", ".csv", ".json", ".html"],
                            file_count="single",
                        )
                        with gr.Row():
                            file_chunk_size = gr.Slider(200, 2000, value=500, step=50, label="分块大小（字符）")
                            file_chunk_overlap = gr.Slider(0, 500, value=100, step=20, label="块重叠")
                        upload_btn = gr.Button("📤 上传到知识库", variant="primary", elem_classes="primary")
                    with gr.Column(scale=1):
                        file_status = gr.Markdown("选择文件后点击上传")

            with gr.Accordion("✏️ 粘贴文本"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            placeholder="把你想让助手记住的内容粘贴到这里...\n比如：个人笔记、日记、材料、或者任何你想让ta知道的事情",
                            lines=10, label="文本内容"
                        )
                        text_source = gr.Textbox(
                            placeholder="来源名称（比如：我的日记、学习笔记）",
                            label="来源",
                            value="手动输入",
                        )
                        text_btn = gr.Button("📤 添加到知识库", variant="primary", elem_classes="primary")
                    with gr.Column(scale=1):
                        text_status = gr.Markdown("粘贴文本后点击添加")

            with gr.Accordion("📊 知识库概览"):
                kb_stats = gr.Markdown("📚 加载中...")
                refresh_kb_btn = gr.Button("🔄 刷新统计", variant="secondary", elem_classes="secondary")
                clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="secondary", elem_classes="secondary")

        # ---- 底部 ----
        gr.HTML("""
        <div class="footer">
            <span class="dot"></span><span class="status">localrag v1.0</span> &nbsp;|&nbsp;
            完全本地 · 零联网 · 零配置 · 你的数据只属于你
        </div>
        """)

        # ======== 事件绑定 ========

        # 手动初始化按钮（备用，正常启动已自动初始化）
        init_btn.click(fn=init_system, outputs=[init_log])

        # 主题切换
        theme_selector.change(fn=lambda k: build_theme_css(k), inputs=[theme_selector], outputs=[theme_css])

        # 聊天
        def respond(message, history, temp, tp, mt, tk, enable_kb, ctx):
            for response in chat_fn(message, history, temp, tp, mt, tk, enable_kb, ctx):
                yield response
            _auto_save()  # 聊完自动保存

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, temperature, top_p, max_tokens, top_k, knowledge_toggle, context_window],
            outputs=[chatbot],
        ).then(lambda: "", None, [msg_input]).then(fn=_update_conv_dropdown, outputs=[conv_dropdown])

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, temperature, top_p, max_tokens, top_k, knowledge_toggle, context_window],
            outputs=[chatbot],
        ).then(lambda: "", None, [msg_input]).then(fn=_update_conv_dropdown, outputs=[conv_dropdown])

        # 清空对话
        clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

        # 人格切换 — 下拉框选中即切换，不需要额外点按钮
        def on_persona_change(persona_key):
            msg, clear_chatbot = switch_persona(persona_key)
            info = get_persona_info(persona_key)
            return msg, clear_chatbot, info
        persona_dropdown.change(fn=on_persona_change, inputs=[persona_dropdown], outputs=[init_log, chatbot, persona_info])
        persona_btn.click(fn=switch_persona, inputs=[persona_dropdown], outputs=[init_log, chatbot])

        # 对话管理
        def on_page_load():
            return _update_conv_dropdown()
        demo.load(fn=on_page_load, outputs=[conv_dropdown])

        new_conv_btn.click(fn=new_conversation, outputs=[chatbot, conv_dropdown])
        load_conv_btn.click(fn=load_conversation, inputs=[conv_dropdown], outputs=[chatbot, conv_dropdown])
        del_conv_btn.click(fn=delete_conversation, inputs=[conv_dropdown], outputs=[conv_dropdown])

        # 文件上传
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload, file_chunk_size, file_chunk_overlap],
            outputs=[file_status, kb_stats],
        )

        # 文本添加
        text_btn.click(
            fn=add_text_knowledge,
            inputs=[text_input, text_source],
            outputs=[text_status, kb_stats],
        )

        # 知识库管理
        refresh_kb_btn.click(fn=_knowledge_stats, outputs=[kb_stats])
        clear_kb_btn.click(fn=clear_knowledge, outputs=[init_log, kb_stats])

    return demo


def main():
    """启动 Web UI — 启动前自动初始化所有组件"""
    config = get_config()

    # ====== 先初始化，再启动 UI ======
    p = print  # shortcut with flush
    p()
    p("=" * 55, flush=True)
    p("  localrag — 离线 AI 聊天助手 v1.0", flush=True)
    p("=" * 55, flush=True)
    p()
    p("[1/3] 加载嵌入模型...", flush=True)
    try:
        _state.embedder = get_embedder()
        _state.embedder.load()
        p("      ✅ 嵌入模型就绪", flush=True)
    except Exception as e:
        p(f"      ❌ 嵌入模型失败: {e}", flush=True)

    p("[2/3] 初始化知识库...", flush=True)
    try:
        _state.vectordb = get_vectordb()
        _state.vectordb.load()
        p(f"      ✅ 知识库就绪 ({_state.vectordb.count} 块)", flush=True)
    except Exception as e:
        p(f"      ❌ 知识库失败: {e}", flush=True)

    p("[3/3] 加载聊天模型...", flush=True)
    try:
        _state.engine = get_engine()
        model_path = Path("models") / config.model.chat_model_file
        if model_path.exists():
            _state.engine.load(model_path)
            _state.engine.set_personality(_state.current_persona)
            _state.is_ready = True
            p("      ✅ 聊天模型就绪", flush=True)
            # 预加载人格私有知识库
            kb_msg = _build_persona_kb(_state.current_persona)
            if kb_msg:
                p(f"      {kb_msg}", flush=True)
        else:
            p("      ⚠️  聊天模型文件不存在", flush=True)
    except Exception as e:
        p(f"      ❌ 聊天模型失败: {e}", flush=True)

    p()
    if _state.is_ready:
        p("  🎉 一切就绪！开始聊天吧！", flush=True)
    else:
        p("  ⚠️  部分组件未就绪", flush=True)
    p()
    p(f"  浏览器打开: http://localhost:{config.ui.server_port}", flush=True)
    p("  Ctrl+C 停止", flush=True)
    p("=" * 55, flush=True)
    p()

    # ====== 启动 UI（已经初始化好了）======
    demo = create_ui()
    demo.queue(max_size=20)

    demo.launch(
        server_port=config.ui.server_port,
        share=config.ui.share,
        inbrowser=config.ui.inbrowser,
    )
