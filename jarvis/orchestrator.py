"""Orchestrator：把 AudioPipeline / ASR / Brain / TTS 串成全双工流水线 + VAD-based barge-in。

状态机：
  LISTENING ──ASR端点──> THINKING ──首句就绪──> SPEAKING ──播完──> LISTENING
                         ▲                       │
                         └──── barge-in ─────────┘  (VAD 检出人声→停TTS→停LLM→回听)

barge-in：SPEAKING 期间，AudioPipeline 的 silero VAD 持续工作；
orchestrator 轮询 pipeline.is_speech_recent(window, min_speech)，
命中即判定用户插话。比 energy 阈值可靠：silero 能区分人声与噪声/残留回声。
配合 AEC（far-end reference 来自 TTS 输出），回声被先消除，VAD 再判。
"""

from __future__ import annotations

import threading
import time
from enum import Enum

from audio_pipeline import AudioPipeline
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
        self._bargein = threading.Event()
        self._state = State.LISTENING
        self._state_lock = threading.Lock()

        # 统一音频管线：麦克风采集 + AEC + VAD
        self._pipeline = AudioPipeline()

        # TTS：注入 pipeline，播放时同步喂 far-end reference
        self._tts = TTS(pipeline=self._pipeline)

        # Brain：每生成一句就喂给 TTS；should_stop 查打断标志
        self._brain = Brain(
            on_sentence=self._on_sentence,
            should_stop=self._bargein.is_set,
        )

        # ASR：从 pipeline 拿干净音频
        self._asr = ASR(
            pipeline=self._pipeline,
            on_endpoint=self._on_endpoint,
            on_partial=self._on_partial,
        )

        # barge-in 监视线程
        self._watcher: threading.Thread | None = None
        self._running = False

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        self._running = True
        # 先启 pipeline（采集），再启 TTS/ASR（消费）
        self._pipeline.start()
        self._tts.start()
        self._asr.start()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()
        self._print("[Jarvis] 就绪（VAD+AEC 全双工），请说话…（Ctrl+C 退出）")

    def stop(self) -> None:
        self._running = False
        self._bargein.set()
        self._asr.stop()
        self._tts.shutdown()
        self._pipeline.stop()
        if self._watcher:
            self._watcher.join(timeout=2.0)

    # -------------------- ASR 回调 --------------------
    def _on_partial(self, text: str) -> None:
        if self._get_state() == State.LISTENING:
            self._print(f"\r  你说: {text}", end="", flush=True)

    def _on_endpoint(self, text: str) -> None:
        self._print(f"\r  你说: {text}" + " " * 10)
        self._bargein.clear()
        self._set_state(State.THINKING)

        gen_thread = threading.Thread(
            target=self._generate, args=(text,), daemon=True
        )
        gen_thread.start()

    # -------------------- 生成与播放 --------------------
    def _generate(self, text: str) -> None:
        try:
            self._brain.respond(text)
        finally:
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
        deadline = time.time() + timeout
        while time.time() < deadline and self._running:
            if self._tts._queue.empty() and not self._bargein.is_set():
                return
            time.sleep(0.05)

    # -------------------- Barge-in 监视（VAD-based）--------------------
    def _watch_loop(self) -> None:
        while self._running:
            if self._get_state() == State.SPEAKING and cfg.bargein_enabled:
                self._check_bargein()
            time.sleep(0.02)

    def _check_bargein(self) -> None:
        """VAD 在最近 window 内检出累计 ≥ min_speech 的语音段 → 判定插话。

        比 energy 阈值可靠：silero 能区分人声与噪声/残留回声。
        仍建议配合 AEC；纯 VAD 无 AEC 时残留回声可能误触发。
        """
        if not self._pipeline.is_speech_recent(
            cfg.bargein_window_ms, cfg.bargein_min_speech_ms
        ):
            return
        # 命中：打断
        self._print("[Jarvis] (VAD 打断)")
        self._bargein.set()
        self._tts.stop()              # 瞬时静音 + 清队列
        self._pipeline.reset_aec()     # 清 far 缓冲 + 重置 AEC 自适应
        self._brain.reset_context()

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
