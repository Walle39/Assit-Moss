"""Orchestrator：把 ASR / Brain / TTS 串成一条流水线 + barge-in 协调。

状态机：
  LISTENING ──ASR端点──> THINKING ──首句就绪──> SPEAKING ──播完──> LISTENING
                         ▲                       │
                         └──── barge-in ─────────┘  (用户插话→停TTS→停LLM→回听)

barge-in 检测：SPEAKING 期间轮询 asr.current_energy_db，
超过阈值且持续 >= bargein_min_ms 即判定插话。
"""

from __future__ import annotations

import threading
import time
from enum import Enum

from asr import ASR
from brain import Brain
from tts import TTS
from config import cfg


class State(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self) -> None:
        self._bargein = threading.Event()  # 由监听线程 set，brain/tts 读
        self._state = State.LISTENING
        self._state_lock = threading.Lock()

        # TTS：句子进来就播
        self._tts = TTS()

        # Brain：每生成一句就喂给 TTS；should_stop 查打断标志
        self._brain = Brain(
            on_sentence=self._on_sentence,
            should_stop=self._bargein.is_set,
        )

        # ASR：端点出整句 → 处理
        self._asr = ASR(
            on_endpoint=self._on_endpoint,
            on_partial=self._on_partial,
        )

        # barge-in 监视线程
        self._watcher: threading.Thread | None = None
        self._running = False

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        self._running = True
        self._tts.start()
        self._asr.start()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()
        self._print("[Jarvis] 就绪，请说话…（Ctrl+C 退出）")

    def stop(self) -> None:
        self._running = False
        self._bargein.set()
        self._asr.stop()
        self._tts.shutdown()
        if self._watcher:
            self._watcher.join(timeout=2.0)

    # -------------------- ASR 回调 --------------------
    def _on_partial(self, text: str) -> None:
        # 只在听音时打印转写
        if self._get_state() == State.LISTENING:
            self._print(f"\r  你说: {text}", end="", flush=True)

    def _on_endpoint(self, text: str) -> None:
        self._print(f"\r  你说: {text}" + " " * 10)
        # 清掉可能的遗留打断标志
        self._bargein.clear()
        self._set_state(State.THINKING)

        # 在独立线程跑生成，主循环继续；barge-in 监视并行
        gen_thread = threading.Thread(
            target=self._generate, args=(text,), daemon=True
        )
        gen_thread.start()

    # -------------------- 生成与播放 --------------------
    def _generate(self, text: str) -> None:
        try:
            self._brain.respond(text)
        finally:
            # 生成结束（或被打断）：若没被 bar-ge-in，等 TTS 播完再回听
            if not self._bargein.is_set():
                self._wait_speaking_done()
            self._bargein.clear()
            self._set_state(State.LISTENING)
            self._print("[Jarvis] 请说…")

    def _on_sentence(self, seg: str) -> None:
        """Brain 每产出一句话：进 TTS 队列并切到 SPEAKING。"""
        self._print(f"  Jarvis: {seg}")
        if self._bargein.is_set():
            return
        self._set_state(State.SPEAKING)
        self._tts.speak(seg)

    def _wait_speaking_done(self, timeout: float = 30.0) -> None:
        """等 TTS 队列排空。简单轮询即可。"""
        deadline = time.time() + timeout
        while time.time() < deadline and self._running:
            if self._tts._queue.empty() and not self._bargein.is_set():
                return
            time.sleep(0.05)

    # -------------------- Barge-in 监视 --------------------
    def _watch_loop(self) -> None:
        while self._running:
            if self._get_state() == State.SPEAKING and cfg.bargein_enabled:
                self._check_bargein()
            time.sleep(0.02)

    def _check_bargein(self) -> None:
        energy = self._asr.current_energy_db
        if energy < cfg.bargein_energy_db:
            return
        # 持续超阈值达 min_ms 才算插话，滤掉咳嗽/背景噪声
        held = 0.0
        while self._running and self._get_state() == State.SPEAKING:
            e = self._asr.current_energy_db
            if e >= cfg.bargein_energy_db:
                held += 0.02
            else:
                return  # 能量回落，不算插话
            if held * 1000 >= cfg.bargein_min_ms:
                self._print("[Jarvis] (打断)")
                self._bargein.set()
                self._tts.stop()              # 瞬时静音
                self._brain.reset_context()   # 丢弃不完整回复
                return
            time.sleep(0.02)

    # -------------------- 工具 --------------------
    def _set_state(self, s: State) -> None:
        with self._state_lock:
            self._state = s

    def _get_state(self) -> State:
        with self._state_lock:
            return self._state

    @staticmethod
    def _print(msg: str, **kw) -> None:
        print(msg, **kw)
