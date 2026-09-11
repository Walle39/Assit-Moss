"""Brain：MiniCPM5-1B 流式生成 + 句边界切分 + 可中断。

不依赖 Ollama：直接用 llama-cpp-python 加载 GGUF。

安装 llama-cpp-python（按平台选 wheel，避免本地编译）：
  pip install llama-cpp-python
  * macOS Apple Silicon 预编译已带 Metal，可加 CMAKE_ARGS="-DGGML_METAL=on"
  * NVIDIA CUDA：预编译 wheel 默认含 CUDA；如需自编译：
      CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
  权重：https://huggingface.co/openbmb/MiniCPM5-1B-GGUF（Q4_K_M ~0.5GB）

生成是流式的：每出一个 token，检查 should_stop()，并把累积文本
按句号/问号/换行切成段，遇到边界就调 on_sentence(segment)。
"""

from __future__ import annotations

import re
import threading
from typing import Callable

from llama_cpp import Llama

from config import cfg

# MiniCPM5 用 ChatML 模板；create_chat_completion 会自动套用 GGUF 内嵌模板。
# 若该 GGUF 未内嵌模板，会抛错，届时改用 _build_manual_prompt() 手动拼。
SENT_BOUNDARY = re.compile(r"[。！？!?;；\n]+")


class Brain:
    """流式 LLM 大脑。维护对话历史，按句输出，支持 barge-in 中断。"""

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self._llm = Llama(
            model_path=cfg.llm_model,
            n_ctx=cfg.llm_n_ctx,
            n_gpu_layers=cfg.llm_n_gpu_layers,
            n_threads=cfg.llm_n_threads,
            verbose=False,
        )
        self._on_sentence = on_sentence
        self._should_stop = should_stop
        self._history: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
        self._gen_lock = threading.Lock()

    # -------------------- 对外接口 --------------------
    def respond(self, user_text: str) -> None:
        """处理一句用户输入，流式按句回调回复。阻塞调用，被 should_stop 中断。"""
        self._history.append({"role": "user", "content": user_text})

        with self._gen_lock:
            chunks = self._stream_chat()

            buf = ""
            full_reply = ""
            for token in chunks:
                if self._should_stop():
                    break
                token = token or ""
                buf += token
                full_reply += token

                # 按句切分：遇到边界就送出一段
                m = SENT_BOUNDARY.search(buf)
                while m:
                    seg = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if seg:
                        self._on_sentence(seg)
                    m = SENT_BOUNDARY.search(buf)

            # 收尾：把剩余 buffer 送出
            tail = buf.strip()
            if tail and not self._should_stop():
                self._on_sentence(tail)

        # 记录完整回复到历史（被打断时也记录已生成部分，保持上下文连贯）
        if full_reply.strip():
            self._history.append({"role": "assistant", "content": full_reply.strip()})

        # 限制历史长度，防止上下文膨胀
        self._trim_history()

    def reset_context(self) -> None:
        """barge-in 后可调用，清掉当前不完整轮次，只留 system + 历史。"""
        # 移除末尾若是未完成 assistant 项
        while self._history and self._history[-1]["role"] == "assistant":
            self._history.pop()

    # -------------------- 内部 --------------------
    def _stream_chat(self):
        """优先用 create_chat_completion（自动套模板）。失败则手动拼 prompt。"""
        try:
            stream = self._llm.create_chat_completion(
                messages=self._history,
                stream=True,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
                stop=["<|im_end|>"],
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception:
            # 降级：手动构造 ChatML prompt
            prompt = self._build_manual_prompt()
            for token in self._llm(
                prompt,
                stream=True,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
                stop=["<|im_end|>"],
            ):
                yield token.get("choices", [{}])[0].get("text", "")

    def _build_manual_prompt(self) -> str:
        """MiniCPM5 ChatML 风格 prompt（GGUF 未内嵌模板时的兜底）。"""
        parts = []
        for m in self._history:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _trim_history(self, max_turns: int = 12) -> None:
        """保留 system + 最近 max_turns 条消息。"""
        system = [m for m in self._history if m["role"] == "system"]
        rest = [m for m in self._history if m["role"] != "system"]
        rest = rest[-max_turns:]
        self._history = system + rest
