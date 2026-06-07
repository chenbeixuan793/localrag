"""LLM 聊天引擎

策略：优先使用 llama-cpp-python（进程内推理），
      如果不可用，自动下载 llama-server 并通过 HTTP API 调用。

两种模式都支持流式输出和上下文管理。
"""

import os
import sys
import json
import time
import atexit
import subprocess
import signal
import tempfile
from pathlib import Path
from typing import Optional, Callable
from threading import Lock

import requests

from .config import get_config, AppConfig, MODELS_DIR
from .personality import build_system_prompt, get_preset

# 全局引擎单例
_engine = None
_engine_lock = Lock()

def _find_llama_server() -> Optional[Path]:
    """查找 llama-server 可执行文件"""
    # 1. 先看 models 目录
    for name in ["llama-server.exe", "llama-server"]:
        p = MODELS_DIR / name
        if p.exists():
            return p
    # 2. 系统 PATH
    import shutil
    for name in ["llama-server.exe", "llama-server"]:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


class ChatEngine:
    """本地 LLM 对话引擎（支持两种模式）"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._llm = None            # llama-cpp-python 实例
        self._server_proc = None    # llama-server 子进程
        self._server_url = None     # llama-server HTTP 地址
        self._use_server = False    # 是否使用 server 模式
        self._model_path: Optional[Path] = None
        self._conversation_history: list[dict] = []
        self._system_prompt: str = ""
        self._persona_name: str = ""
        self._max_history_turns: int = 20
        self._context_window: int = 8192  # 有效上下文窗口（token 数），UI 可调

    # ======== 加载 ========

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None or self._server_proc is not None

    def load(self, model_path: Optional[Path] = None) -> bool:
        """加载聊天模型（自动选择最佳方式）"""
        if model_path is None:
            model_path = MODELS_DIR / self.config.model.chat_model_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先运行: localrag download\n"
                f"或手动下载 GGUF 模型放到: {model_path}"
            )

        self._model_path = model_path

        # 方案 A: 尝试 llama-cpp-python
        try:
            from llama_cpp import Llama
            print(f"🧠 加载聊天模型 (llama-cpp-python): {model_path.name}")
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.model.chat_n_ctx,
                n_threads=self.config.model.chat_n_threads,
                n_batch=self.config.model.chat_n_batch,
                verbose=False,
            )
            self._use_server = False
            print("✅ 聊天模型就绪! (进程内模式)")
            return True
        except ImportError:
            print("⚠️  llama-cpp-python 未安装，使用 llama-server 模式...")
        except Exception as e:
            print(f"⚠️  llama-cpp-python 加载失败: {e}")
            print("   切换为 llama-server 模式...")

        # 方案 B: llama-server 子进程
        return self._load_with_server(model_path)

    def _load_with_server(self, model_path: Path) -> bool:
        """使用 llama-server 子进程加载模型"""
        server_bin = _find_llama_server()

        if server_bin is None:
            print("[engine] llama-server not found, trying auto-download...")
            from .download import download_llama_server
            download_llama_server()
            server_bin = _find_llama_server()

        if server_bin is None:
            raise RuntimeError(
                "无法获取 llama-server。请手动下载:\n"
                "  https://github.com/ggerganov/llama.cpp/releases/latest\n"
                f"  将 llama-server.exe 放到: {MODELS_DIR}"
            )

        # 找个空闲端口
        import socket
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        self._server_url = f"http://127.0.0.1:{port}"

        print(f"🧠 启动 llama-server (端口 {port}): {model_path.name}")

        cmd = [
            str(server_bin),
            "-m", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(self.config.model.chat_n_ctx),
            "-t", str(self.config.model.chat_n_threads),
            "-b", str(self.config.model.chat_n_batch),
            "--no-mmap",
        ]

        try:
            self._server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(f"启动 llama-server 失败: {e}")

        # 等待服务就绪
        print("   等待 llama-server 启动...")
        for i in range(60):
            time.sleep(1)
            if self._server_proc.poll() is not None:
                stderr = self._server_proc.stderr.read() if self._server_proc.stderr else ""
                raise RuntimeError(f"llama-server 异常退出: {stderr}")
            try:
                resp = requests.get(f"{self._server_url}/health", timeout=1)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
        else:
            raise RuntimeError("llama-server 启动超时（60秒）")

        self._use_server = True
        atexit.register(self._cleanup_server)
        print("✅ 聊天模型就绪! (llama-server 模式)")
        return True

    def _cleanup_server(self):
        """清理子进程"""
        if self._server_proc:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=5)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            self._server_proc = None

    # ======== 人格 ========

    def set_personality(self, persona_key: str = "warm-friend", user_name: Optional[str] = None):
        preset = get_preset(persona_key)
        self._persona_name = preset["name"]
        self._system_prompt = build_system_prompt(
            name=preset["name"],
            persona_slug=persona_key,
            user_name=user_name,
            extra_traits=preset.get("extra_traits"),
        )
        return preset

    # ======== 对话 ========

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数：中文 ≈ 1 token/字，英文 ≈ 0.75 token/词"""
        return len(text)  # 保守估算，1字符≈1token

    def _build_messages(self, user_message: str, knowledge: Optional[str] = None) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        if knowledge:
            rag_note = (
                "\n\n【关于当前话题的记忆/知识 — 请自然地融入对话，"
                "不要说'根据知识库'，而是说'我记得'、'之前我们聊到过'等人类说法】\n"
                f"{knowledge}"
            )
            messages[0]["content"] += rag_note

        # 从旧到新逐步加历史，直到接近上下文窗口上限
        sys_tokens = self._estimate_tokens(messages[0]["content"])
        reserve_for_reply = min(self.config.model.max_tokens, 2048)
        budget = self._context_window - sys_tokens - self._estimate_tokens(user_message) - reserve_for_reply

        history = self._conversation_history[:]
        recent = []
        used = 0
        for msg in reversed(history):
            t = self._estimate_tokens(msg.get("content", ""))
            if used + t > budget:
                break
            recent.insert(0, msg)
            used += t

        messages.extend(recent)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _chat_llama_cpp(self, messages: list[dict]) -> str:
        """使用 llama-cpp-python 推理"""
        prompt = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        full = ""
        for output in self._llm(
            prompt,
            max_tokens=self.config.model.max_tokens,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            top_k=self.config.model.top_k,
            repeat_penalty=self.config.model.repeat_penalty,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            stream=True,
        ):
            choices = output.get("choices", [])
            if choices:
                full += choices[0].get("text", "")

        return full.replace("<|im_end|>", "").replace("<|im_start|>", "")

    def _chat_server(self, messages: list[dict]) -> str:
        """使用 llama-server HTTP API 推理"""
        resp = requests.post(
            f"{self._server_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "max_tokens": self.config.model.max_tokens,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat(self, message: str, knowledge: Optional[str] = None,
             stream: bool = True, on_token: Optional[Callable[[str], None]] = None) -> str:
        """对话接口"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        messages = self._build_messages(message, knowledge)

        if self._use_server:
            full_response = self._chat_server(messages)
        else:
            full_response = self._chat_llama_cpp(messages)

        # 模拟流式逐 token
        if on_token:
            for i in range(0, len(full_response), 3):
                on_token(full_response[i:i+3])

        # 保存历史
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({"role": "assistant", "content": full_response})
        if len(self._conversation_history) > self._max_history_turns * 2:
            self._conversation_history = self._conversation_history[-(self._max_history_turns * 2):]

        return full_response

    def clear_history(self):
        self._conversation_history = []

    def get_history(self) -> list[dict]:
        return self._conversation_history

    def set_context_window(self, n_ctx: int):
        """设置有效上下文窗口大小（UI 滑块调用）"""
        self._context_window = max(2048, min(n_ctx, 32768))

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k == "context_window":
                self.set_context_window(v)
            elif hasattr(self.config.model, k):
                setattr(self.config.model, k, v)


def get_engine() -> ChatEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ChatEngine()
        return _engine
