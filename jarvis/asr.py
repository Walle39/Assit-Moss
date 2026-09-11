"""ASR：sherpa-onnx 流式语音识别 + 端点检测。

模型下载：
  https://github.com/k2-fsa/sherpa-onnx/releases（找 asr-models）
  推荐双语：sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  里面包含 encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt / bpe.model

设计要点：
- 不再自己采集麦克风：从 AudioPipeline 拿 AEC 后的干净音频。
- AudioPipeline 负责 VAD + AEC，这里只做 ASR + 端点。
- 一条消费线程从 pipeline.clean_audio_q 取块，喂给 OnlineRecognizer。
- partial 实时回调，endpoint 整句回调。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import sherpa_onnx

from config import cfg
from audio_pipeline import AudioPipeline


class ASR:
    """流式 ASR + 端点。从 AudioPipeline 消费干净音频。"""

    def __init__(
        self,
        pipeline: AudioPipeline,
        on_endpoint: Callable[[str], None],
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._on_endpoint = on_endpoint
        self._on_partial = on_partial or (lambda _: None)

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

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -------------------- 消费循环 --------------------
    def _consume_loop(self) -> None:
        prev_partial = ""
        q = self._pipeline.clean_audio_q
        while self._running:
            if not q:
                time.sleep(0.01)
                continue
            samples = q.popleft()
            # 喂给识别器（已是 AEC 后的干净音频）
            self._stream.accept_waveform(cfg.audio_sample_rate, samples)
            self._recognizer.decode_stream(self._stream)

            # partial
            partial = self._recognizer.get_result(self._stream)
            if partial and partial != prev_partial:
                prev_partial = partial
                self._on_partial(partial)

            # 端点
            if self._recognizer.is_endpoint(self._stream):
                text = self._recognizer.get_result(self._stream).strip()
                self._recognizer.reset(self._stream)
                prev_partial = ""
                if text:
                    self._on_endpoint(text)
