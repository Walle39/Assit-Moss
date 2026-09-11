"""TTS：MOSS-TTS-Nano 句级流式合成 + 播放队列 + barge-in 打断。

安装 MOSS-TTS-Nano（从源码，非 pypi）：
  git clone https://github.com/OpenMOSS/MOSS-TTS-Nano
  cd MOSS-TTS-Nano
  pip install -r requirements.txt
  pip install -e .          # 注册 moss-tts-nano 命令

本骨架默认走 `moss-tts-nano` CLI（每句一次子进程）——最可靠、即装即跑。
代价是每句有几百 ms 启动开销。要更低延迟，把 _synthesize() 换成
进程内加载 ONNX 模型（MOSS-TTS-Nano-100M-ONNX）的常驻实现即可，
接口形状不变：text -> (numpy float32, sample_rate)。

barge-in：
- 句子进播放队列；播放线程用 sounddevice.OutputStream 按块拉取。
- stop() 立即清空队列并停止输出流，声音瞬时静音。
"""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
from typing import Callable

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except OSError as e:
    sd = None
    _SD_ERR = e
else:
    _SD_ERR = None

from config import cfg
from audio_pipeline import AudioPipeline  # for far-end reference

# 哨兵：放进队列表示"后面没有了"
_DONE = object()


class TTS:
    def __init__(self, pipeline: AudioPipeline | None = None) -> None:
        if sd is None:
            raise RuntimeError(
                f"sounddevice 初始化失败（缺 PortAudio？）：{_SD_ERR}"
            )
        self._pipeline = pipeline  # 播放时同步喂 far-end reference
        self._queue: queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._player: threading.Thread | None = None
        self._stream = None  # sounddevice OutputStream

    # -------------------- 生命周期 --------------------
    def start(self) -> None:
        if self._player and self._player.is_alive():
            return
        self._stop_flag.clear()
        self._player = threading.Thread(target=self._play_loop, daemon=True)
        self._player.start()

    def stop(self) -> None:
        """barge-in：清队列、停播放、瞬时静音。"""
        self._drain_queue()
        self._stop_flag.set()
        try:
            if self._stream is not None:
                self._stream.stop()
        except Exception:
            pass

    def shutdown(self) -> None:
        """彻底结束（程序退出时）。"""
        self._queue.put(_DONE)
        self._stop_flag.set()
        if self._player:
            self._player.join(timeout=2.0)
            self._player = None

    # -------------------- 喂句子 --------------------
    def speak(self, text: str) -> None:
        """把一句文本交给 TTS：异步合成后入播放队列。"""
        if not text.strip():
            return
        # 合成在独立线程，避免阻塞 LLM 流
        t = threading.Thread(
            target=self._produce, args=(text,), daemon=True
        )
        t.start()

    # -------------------- 合成（可替换）--------------------
    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """text -> (float32 mono, sample_rate)。
        默认实现：调用 moss-tts-nano CLI，读回 wav。"""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        )
        tmp.close()
        cmd = [
            "moss-tts-nano", "generate",
            "--text", text,
            "--output", tmp.name,
        ]
        # 语音克隆：有参考音频就克隆，否则用预设音色
        if cfg.tts_ref_audio and os.path.exists(cfg.tts_ref_audio):
            cmd += ["--prompt-speech", cfg.tts_ref_audio]
        else:
            cmd += ["--voice", cfg.tts_voice]

        subprocess.run(cmd, check=True, capture_output=True)
        data, sr = sf.read(tmp.name, dtype="float32")
        os.unlink(tmp.name)

        # 统一成 mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    def _produce(self, text: str) -> None:
        try:
            audio, sr = self._synthesize(text)
        except Exception as e:
            # 合成失败不该卡住整条流水线
            print(f"[TTS] 合成失败: {e}")
            return
        if self._stop_flag.is_set():
            return
        self._queue.put((audio, sr))

    # -------------------- 播放线程 --------------------
    def _play_loop(self) -> None:
        """从队列取音频块，按顺序播放。被 stop() 后立刻结束当前句。"""
        while not self._stop_flag.is_set():
            item = self._queue.get()
            if item is _DONE:
                break
            audio, sr = item
            if self._stop_flag.is_set():
                break
            self._play(audio, sr)

    def _play(self, audio: np.ndarray, sr: int) -> None:
        try:
            self._stream = sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32"
            )
            self._stream.start()
            # 分块写，便于及时响应 stop()。
            # 同时把每个块喂给 AEC 当 far-end reference ——
            # 这样 AEC 拿到的 reference 与麦克风听到的回声相位一致。
            chunk = 2048
            for i in range(0, len(audio), chunk):
                if self._stop_flag.is_set():
                    break
                block = audio[i : i + chunk]
                self._stream.write(block[:, None])
                # 同步喂 far-end（pipeline 会自动重采样到 ASR 采样率）
                if self._pipeline is not None:
                    self._pipeline.push_far_end(block, sr)
        except Exception as e:
            print(f"[TTS] 播放失败: {e}")
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
