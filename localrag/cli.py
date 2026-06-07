"""CLI 命令行入口 — localrag 命令"""

import sys
import os
from pathlib import Path

# 强制 UTF-8 输出（Windows 兼容）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .config import get_config, MODELS_DIR, KNOWLEDGE_DIR
from .personality import PRESET_PERSONAS

console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    """打印 banner"""
    console.print(Panel.fit(
        "[bold violet]💫 离线 AI 聊天助手[/bold violet]  [dim]v{version}[/dim]\n"
        "[dim]完全本地运行 · 不吃配置 · 不调任何 API · 你的数据只属于你[/dim]".format(version=__version__),
        border_style="violet",
    ))


@click.group()
@click.version_option(__version__, prog_name="localrag")
@click.pass_context
def main(ctx: click.Context):
    """💫 localrag — 离线本地 RAG 聊天助手

    一条命令启动，不吃配置，纯 CPU 运行。
    """
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("\n[yellow]用法:[/yellow] localrag [命令]")
        console.print("\n[bold]常用命令:[/bold]")
        console.print("  [violet]start[/violet]      启动 Web 聊天界面")
        console.print("  [violet]download[/violet]   下载所需模型")
        console.print("  [violet]chat[/violet]       终端聊天模式")
        console.print("  [violet]ingest[/violet]     导入文档到知识库")
        console.print("  [violet]info[/violet]       查看系统信息")
        console.print("  [violet]clear[/violet]      清空知识库")
        console.print("\n[dim]运行 'localrag <命令> --help' 查看更多选项[/dim]")


@main.command()
@click.option("--port", "-p", default=None, type=int, help="服务端口 (默认: 7860)")
@click.option("--no-browser", is_flag=True, help="不自动打开浏览器")
@click.option("--share", is_flag=True, help="生成公网分享链接")
def start(port, no_browser, share):
    """启动 Web 聊天界面"""
    from .app import main as app_main

    if port:
        get_config().ui.server_port = port
    if no_browser:
        get_config().ui.inbrowser = False
    if share:
        get_config().ui.share = True

    app_main()


@main.command()
@click.option("--mirror/--no-mirror", default=True, help="使用 HuggingFace 镜像加速 (默认: 启用)")
def download(mirror):
    """下载所需的聊天模型和嵌入模型"""
    print_banner()
    console.print("\n[bold]📥 下载模型...[/bold]\n")

    from .download import ensure_models

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("检查/下载模型...", total=None)
            chat_path, embed_path = ensure_models(use_mirror=mirror)
            progress.update(task, completed=True)

        console.print(f"\n[green]✅ 模型就绪![/green]")
        console.print(f"   聊天模型: [cyan]{chat_path}[/cyan]")
        console.print(f"\n   现在运行 [violet]localrag start[/violet] 启动聊天助手!")

    except Exception as e:
        console.print(f"\n[red]❌ 下载失败: {e}[/red]")
        console.print("\n[yellow]手动下载:[/yellow]")
        console.print(f"  1. 从 HuggingFace 下载 GGUF 模型文件")
        console.print(f"  2. 放到: {MODELS_DIR}/")
        console.print(f"  3. 重新运行: localrag start")
        sys.exit(1)


@main.command()
@click.option("--persona", "-p", default="warm-friend", type=click.Choice(list(PRESET_PERSONAS.keys())),
              help="选择人格")
def chat(persona):
    """终端聊天模式（纯文本）"""
    print_banner()

    from .engine import ChatEngine
    from .embedder import Embedder
    from .vectordb import VectorDB
    from .ingest import ingest_text

    console.print("\n[dim]初始化中...[/dim]")

    # 初始化
    engine = ChatEngine()
    embedder = Embedder()
    vectordb = VectorDB()

    try:
        embedder.load()
        vectordb.load()
        model_path = MODELS_DIR / get_config().model.chat_model_file
        engine.load(model_path)
        engine.set_personality(persona)
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        sys.exit(1)

    preset = PRESET_PERSONAS[persona]
    console.print(f"\n[green]✅ {preset['display']} 已就绪![/green]")
    console.print("[dim]输入 'exit' 退出, 'clear' 清空对话, 'kb' 查看知识库状态[/dim]\n")

    while True:
        try:
            user_input = click.prompt("你", prompt_suffix=" > ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见! 👋[/dim]")
            break

        if user_input.lower() == "exit":
            console.print("[dim]再见! 👋[/dim]")
            break
        elif user_input.lower() == "clear":
            engine.clear_history()
            continue
        elif user_input.lower() == "kb":
            console.print(f"[dim]知识库文档块: {vectordb.count}[/dim]")
            continue

        # RAG 检索
        knowledge = None
        if vectordb.count > 0:
            q_emb = embedder.embed_query(user_input)
            results = vectordb.search_text(q_emb)
            if results:
                knowledge = "\n\n---\n".join(results)

        # 流式输出
        console.print(f"\n[bold violet]{preset['name']}[/bold violet] > ", end="")
        full = engine.chat(user_input, knowledge=knowledge, stream=False)
        console.print(full)
        console.print()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--chunk-size", "-c", default=None, type=int, help="分块大小")
@click.option("--chunk-overlap", "-o", default=None, type=int, help="块重叠")
@click.option("--recursive/--no-recursive", default=True, help="递归处理子目录")
def ingest(path, chunk_size, chunk_overlap, recursive):
    """导入文件或目录到知识库"""
    print_banner()

    from .embedder import Embedder, get_embedder
    from .vectordb import VectorDB, get_vectordb
    from .ingest import ingest_file, ingest_directory

    console.print("\n[bold]📚 导入知识库...[/bold]\n")

    # 初始化
    embedder = get_embedder()
    embedder.load()
    vectordb = get_vectordb()
    vectordb.load()

    target = Path(path)

    if target.is_file():
        n = ingest_file(target, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        console.print(f"\n[green]✅ 已导入 {target.name}: {n} 个文档块[/green]")
    else:
        n = ingest_directory(target, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                            recursive=recursive)
        console.print(f"\n[green]✅ 已导入整个目录: {n} 个文档块[/green]")

    console.print(f"[dim]知识库总计: {vectordb.count} 个文档块[/dim]")


@main.command()
def info():
    """查看系统信息"""
    print_banner()

    config = get_config()

    # 配置信息
    table = Table(title="系统配置", border_style="violet")
    table.add_column("类别", style="cyan")
    table.add_column("项目", style="dim")
    table.add_column("值")

    table.add_row("聊天模型", "仓库", config.model.chat_model_repo)
    table.add_row("", "文件名", config.model.chat_model_file)
    table.add_row("", "上下文长度", str(config.model.chat_n_ctx))
    table.add_row("", "CPU 线程", str(config.model.chat_n_threads))
    table.add_row("嵌入模型", "仓库", config.model.embed_model_repo)
    table.add_row("RAG", "分块大小", f"{config.rag.chunk_size} 字符")
    table.add_row("", "块重叠", f"{config.rag.chunk_overlap} 字符")
    table.add_row("", "检索 Top-K", str(config.rag.top_k))
    table.add_row("生成参数", "Temperature", str(config.model.temperature))
    table.add_row("", "Top-P", str(config.model.top_p))
    table.add_row("", "Max Tokens", str(config.model.max_tokens))
    table.add_row("UI", "端口", str(config.ui.server_port))

    console.print(table)

    # 模型状态
    console.print("\n[bold]模型状态:[/bold]")
    chat_model = MODELS_DIR / config.model.chat_model_file
    if chat_model.exists():
        size_mb = chat_model.stat().st_size / (1024 * 1024)
        console.print(f"  聊天模型: [green]✅ 已下载[/green] ({size_mb:.0f} MB)")
    else:
        console.print(f"  聊天模型: [red]❌ 未下载[/red]")
        console.print(f"  运行 [violet]localrag download[/violet] 下载")

    console.print(f"  嵌入模型: [green]✅ 首次使用时自动下载[/green]")

    # 知识库状态
    try:
        from .vectordb import get_vectordb
        vectordb = get_vectordb()
        vectordb.load()
        console.print(f"  知识库: [green]✅ 已初始化[/green] ({vectordb.count} 块)")
    except Exception:
        console.print(f"  知识库: [yellow]⚠️ 未初始化[/yellow]")


@main.command()
@click.option("--yes", "-y", is_flag=True, help="跳过确认")
def clear(yes):
    """清空所有数据（知识库 + 对话历史）"""
    print_banner()

    if not yes:
        if not click.confirm("\n[red]⚠️ 确定要清空所有知识库数据吗？此操作不可恢复！[/red]"):
            console.print("[dim]已取消[/dim]")
            return

    from .vectordb import get_vectordb
    try:
        vectordb = get_vectordb()
        vectordb.load()
        vectordb.delete_all()
        console.print("[green]✅ 知识库已清空[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ 清理时出现警告: {e}[/yellow]")


if __name__ == "__main__":
    main()
