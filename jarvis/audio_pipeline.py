"""统一音频管线：麦克风采集 + far-end reference + 可插拔 AEC + Silero VAD。

这是全双工的核心。数据流：

    扬声器播放的 TTS 音频 (far-end, 助手声音)
                  │
                  ▼
    麦克风 (near-end, 含回声+人声) ──▶ AEC ──▶ 干净近端音频
                                                  │
                                                  ├──▶ Silero VAD ──▶ {speech 段}
                                                  │
                                                  └──▶ ASR（只在干净音频上跑）

barge-in：SPEAKING 期间，VAD 在最近 window_ms 内检出任何语音段 → 用户插话。

AEC 关键设计：far-end reference 直接来自 TTS 即将播放的同一份音频，
不依赖系统回环采集（macOS/Linux 配置回环很麻烦）。TTS 调 push_far_end()
把音频块喂进来，AEC 用它做自适应滤波的参考信号。

AEC 后端可插拔 + 自动降级：
  - "webrtc"：webrtc-audio-processing（效果最好，需 pip install）
  - "speex" ：pyspeex 的 MDF AEC（备选）
  - "none"  ：直通，靠系统级 AEC 兜底
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

try:
    import sounddevice as sd
except OSError as e:
    sd = None
    _SD_ERR = e
else:
    _SD_ERR = None

import sherpa_onnx

from config import cfg


# =====================================================================
# AEC 后端：可插拔，统一接口 process(near, far) -> cleaned_near
# =====================================================================

class AecBackend:
    """AEC 后端抽象。所有后端实现这个接口。"""

    name = "abstract"

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """near/far 同长 float32 mono [-1,1]，返回去回声后的 near。"""
        raise NotImplementedError

    def reset(self) -> None:
        """重置自适应滤波器状态（barge-in 或场景切换时调用）。"""
        pass


class PassthroughAec(AecBackend):
    """直通：不做 AEC。靠系统级 AEC 或纯 VAD 兜底。"""

    name = "none"

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        return near


class WebrtcAec(AecBackend):
    """WebRTC AEC3。需要：pip install webrtc-audio-processing

    webrtc-audio-processing 的 Python 绑定提供 AudioProcessing 模块，
    内含 echo_canceller，支持分段处理。本类做轻量封装。
    """

    name = "webrtc"

    def __init__(self) -> None:
        # 延迟导入，缺失时让 _make_backend 捕获并降级
        from webrtc_audio_processing import AudioProcessing  # type: ignore
        self._ap = AudioProcessing(
            echo_canceller=dict(
                enable=True,
                filter_length_ms=cfg.aec_filter_length_ms,
            ),
            highpass_filter=dict(enable=True),
            noise_suppression=dict(enable=True, level=2),
        )
        # webrtc-audio-processing 要求 16kHz 单声道 int16
        self._resample_target = cfg.audio_sample_rate

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        # float32 -> int16
        near_i16 = _f2i16(near)
        far_i16 = _f2i16(far)
        # WebRTC 的 AEC 接口：process_capture(near, reference=far)
        out = self._ap.process_capture(near_i16, reference=far_i16)
        return _i162f(out.astype(np.int16))

    def reset(self) -> None:
        # 重新创建实例以清空自适应状态
        from webrtc_audio_processing import AudioProcessing  # type: ignore
        self._ap = AudioProcessing(
            echo_canceller=dict(enable=True, filter_length_ms=cfg.aec_filter_length_ms),
            highpass_filter=dict(enable=True),
            noise_suppression=dict(enable=True, level=2),
        )


class SpeexAec(AecBackend):
    """Speex MDF AEC。需要：pip install pyspeex

    Speex 的预处理器 + MDF 回声消除，效果中等但好装。
    """

    name = "speex"

    def __init__(self) -> None:
        import pyspeex  # type: ignore
        frame = int(cfg.audio_sample_rate * 0.02)  # 20ms 帧
        self._frame = frame
        self._den = pyspeex.Denoiser(frame)
        self._echo = pyspeex.EchoCanceller(
            frame_size=frame,
            filter_length=int(cfg.audio_sample_rate * cfg.aec_filter_length_ms / 1000),
        )

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        # 按帧处理
        out = np.zeros_like(near, dtype=np.float32)
        f = self._frame
        for i in range(0, len(near), f):
            n = near[i:i + f]
            r = far[i:i + f] if i + f <= len(far) else far[i:]
            if len(n) < f:
                break
            # MDF: echo_cancellation(input, far_end) -> (cleaned, ...)
            cleaned = self._echo.echo_cancellation(_f2i16(n), _f2i16(r))
            cleaned = cleaned[0] if isinstance(cleaned, tuple) else cleaned
            # 降噪
            cleaned = self._den.process(cleaned)
            out[i:i + f] = _i162f(cleaned.astype(np.int16)[:f])
        return out


def _make_backend() -> AecBackend:
    """按 config 选择后端，失败则自动降级。"""
    choice = cfg.aec_backend
    if choice == "webrtc":
        try:
            return WebrtcAec()
        except Exception as e:
            print(f"[AEC] webrtc 不可用（{type(e).__name__}: {e}），尝试降级…")
    if choice == "speex":
        try:
            return SpeexAec()
        except Exception as e:
            print(f"[AEC] speex 不可用（{type(e).__name__}: {e}），降级到 none")
    if choice == "none":
        pass
    print("[AEC] 使用 Passthrough（无软件 AEC）。"
          "请确保系统级 AEC 已开启，否则回声可能误触发 barge-in。")
    return PassthroughAec()


# =====================================================================
# 音频工具
# =====================================================================

def _f2i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def _i162f(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0)


# =====================================================================
# AudioPipeline：对外的统一管线
# =====================================================================

class AudioPipeline:
    """统一音频管线。

    对外接口：
      - start()/stop()            启停
      - clean_audio_q             干净近端音频队列（AEC 后），给 ASR 消费
      - push_far_end(audio)        TTS 播放时把音频喂给 AEC 当 reference
      - is_speech_recent(window_ms, min_speech_ms)  barge-in 查询
    """

    def __init__(self) -> None:
        if sd is None:
            raise RuntimeError(f"sounddevice 不可用：{_SD_ERR}")

        self._running = False
        self._thread: threading.Thread | None = None

        # AEC 后端
        self._aec = _make_backend()

        # VAD：sherpa-onnx 的 silero VoiceActivityDetector
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = cfg.vad_model
        vad_cfg.silero_vad.threshold = cfg.vad_threshold
        vad_cfg.silero_vad.min_silence_duration = cfg.vad_min_silence_ms
        vad_cfg.sample_rate = cfg.audio_sample_rate
        vad_cfg.num_threads = cfg.llm_n_threads
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg)

        # far-end reference 缓冲：与 near-end 对齐
        # 麦克风音频和 far-end 可能有几十 ms 相位差，用环形缓冲吸收
        self._far_buf: deque = deque(maxlen=cfg.audio_sample_rate)  # 最多 1s
        self._far_lock = threading.Lock()

        # 干净音频输出队列（给 ASR）
        self.clean_audio_q: "deque[np.ndarray]" = deque(maxlen=2000)

        # speech 段时间戳（用于 barge-in 回看）
        # 存 (timestamp, duration_ms) 对，barge-in 查窗口内累计 ms
        self._speech_events: deque = deque(maxlen=2000)
        self._ev_lock = threading.Lock()

        self._sr = cfg.audio_sample_rate

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset_aec(self) -> None:
        """barge-in 后调用：清 far 缓冲、重置 AEC 自适应状态。"""
        with self._far_lock:
            self._far_buf.clear()
        self._aec.reset()

    # -------------------- far-end 喂入 --------------------
    def push_far_end(self, audio: np.ndarray, sr: int) -> None:
        """TTS 播放时调用：把即将播放的音频喂给 AEC 当 reference。

        audio: float32 mono。会重采样到 ASR 采样率后入缓冲。
        与麦克风采集同步消费。
        """
        if sr != self._sr:
            # 简单线性重采样（MOSS 48kHz → 16kHz）
            ratio = self._sr / sr
            n = int(len(audio) * ratio)
            idx = np.linspace(0, len(audio) - 1, n)
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        with self._far_lock:
            self._far_buf.extend(audio.tolist())

    # -------------------- barge-in 查询 --------------------
    def is_speech_recent(self, window_ms: int, min_speech_ms: int) -> bool:
        """最近 window_ms 内累计语音段时长是否 >= min_speech_ms。"""
        now = time.time()
        window = window_ms / 1000.0
        total = 0.0
        with self._ev_lock:
            # 从后往前扫
            for ts, dur_ms in reversed(self._speech_events):
                if now - ts > window:
                    break
                total += dur_ms
        return total >= min_speech_ms

    # -------------------- 麦克风采集主循环 --------------------
    def _capture_loop(self) -> None:
        block = cfg.audio_block  # 每块样本数
        # silero VAD 要求每块 512 样本（32ms@16kHz），我们用 512 对齐
        vad_block = 512

        def callback(indata: np.ndarray, frames: int, _t, status):
            if status:
                pass
            samples = indata[:, 0].astype(np.float32) / 32768.0
            self._process_block(samples)

        with sd.InputStream(
            samplerate=self._sr,
            channels=cfg.audio_channels,
            dtype="int16",
            blocksize=vad_block,
            callback=callback,
        ):
            while self._running:
                time.sleep(0.02)

    def _process_block(self, near: np.ndarray) -> None:
        """处理一个 512 样本块：取 far → AEC → 输出 → 喂 VAD。"""
        n = len(near)
        # 取等长 far-end reference；不足则补零（说明此时无 TTS 播放）
        with self._far_lock:
            far = np.zeros(n, dtype=np.float32)
            avail = len(self._far_buf)
            take = min(avail, n)
            if take > 0:
                arr = np.array(
                    [self._far_buf.popleft() for _ in range(take)],
                    dtype=np.float32,
                )
                far[:take] = arr
        # AEC
        try:
            cleaned = self._aec.process(near, far)
        except Exception:
            cleaned = near  # AEC 崩了也不能断流

        # 输出给 ASR
        self.clean_audio_q.append(cleaned)

        # 喂 VAD（silero 要求 512 样本块）
        self._vad.accept_waveform(self._sr, cleaned)
        # 取出已完成的语音段
        while not self._vad.empty():
            seg = self._vad.front  # SpeechSegment: start, samples
            self._vad.flush()
            dur_ms = int(len(seg.samples) / self._sr * 1000)
            with self._ev_lock:
                self._speech_events.append((time.time(), dur_ms))
