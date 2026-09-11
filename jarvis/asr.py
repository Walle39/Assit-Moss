"""ASR：sherpa-onnx 流式语音识别 + 端点检测 + barge-in 能量监听。

模型下载：
  https://github.com/k2-fsa/sherpa-onnx/releases（找 asr-models）
  推荐双语：sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  里面包含 encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt / bpe.model

设计要点：
- 一条麦克风线程持续采集 16kHz mono，喂给 OnlineRecognizer。
- decode_stream() 边收边出 partial；is_endpoint() 判定整句结束 → 回调。
- 同时维护 current_energy(dBFS)，供 orchestrator 在 TTS 播放时做 barge-in。
- 进阶：可把能量 barge-in 换成 silero VAD（sherpa-onnx.VoiceActivityDetector）。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

try:
    import sounddevice as sd
except OSError as e:  # PortAudio 缺失
    sd = None
    _SD_ERR = e
else:
    _SD_ERR = None

import sherpa_onnx

from config import cfg


class ASR:
    """流式 ASR + 端点 + 能量监听。回调都在内部线程触发，注意线程安全。"""

    def __init__(
        self,
        on_endpoint: Callable[[str], None],
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        if sd is None:
            raise RuntimeError(
                f"sounddevice 初始化失败（缺 PortAudio？）：{_SD_ERR}"
            )

        self._on_endpoint = on_endpoint
        self._on_partial = on_partial or (lambda _: None)

        # ---- 识别器 ----
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=cfg.asr_encoder,
            decoder=cfg.asr_decoder,
            joiner=cfg.asr_joiner,
            tokens=cfg.asr_tokens,
            bpe_vocab=cfg.asr_bpe_vocab or None,
            endpoint_config=sherpa_onnx.EndpointConfig(
                rule1=sherpa_onnx.EndpointRule(
                    must_contain_nonsilence=False,
                    min_trailing_silence=cfg.ep_min_trailing_silence,
                    min_utterance_length=cfg.ep_min_utterance_length,
                ),
                rule2=sherpa_onnx.EndpointRule(
                    must_contain_nonsilence=True,
                    min_trailing_silence=cfg.ep_min_trailing_silence,
                    min_utterance_length=0.0,
                ),
                rule3=sherpa_onnx.EndpointRule(
                    must_contain_nonsilence=False,
                    min_trailing_silence=0.0,
                    min_utterance_length=20.0,  # 单句最长 20s 强制断
                ),
            ),
            enable_external_buffer=True,
            num_threads=cfg.llm_n_threads,
        )
        self._stream = self._recognizer.create_stream()

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # barge-in 能量监听
        self.current_energy_db: float = -120.0  # 暴露给 orchestrator 轮询

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -------------------- 主循环 --------------------
    def _loop(self) -> None:
        def callback(indata: np.ndarray, frames: int, _time, status):
            # sounddevice 给的是 (frames, channels) int16
            if status:
                pass  # 可记录 status.input_overflow 等
            # int16 -> float32 [-1,1]
            samples = indata[:, 0].astype(np.float32) / 32768.0
            self.current_energy_db = _rms_db(samples)
            self._stream.accept_waveform(cfg.audio_sample_rate, samples)
            self._recognizer.decode_stream(self._stream)

        with sd.InputStream(
            samplerate=cfg.audio_sample_rate,
            channels=cfg.audio_channels,
            dtype="int16",
            blocksize=cfg.audio_block,
            callback=callback,
        ):
            prev_partial = ""
            while self._running:
                # partial 文本
                partial = self._recognizer.get_result(self._stream)
                if partial and partial != prev_partial:
                    prev_partial = partial
                    self._on_partial(partial)

                # 端点判定
                if self._recognizer.is_endpoint(self._stream):
                    text = self._recognizer.get_result(self._stream).strip()
                    # 重置流以接收下一句
                    self._recognizer.reset(self._stream)
                    prev_partial = ""
                    if text:
                        self._on_endpoint(text)

                # 轻量让出 CPU（音频回调是异步的，这里只做轮询）
                time.sleep(0.02)


def _rms_db(samples: np.ndarray) -> float:
    """计算 dBFS 能量。空输入返回 -120。"""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < 1e-7:
        return -120.0
    return 20.0 * float(np.log10(rms))
