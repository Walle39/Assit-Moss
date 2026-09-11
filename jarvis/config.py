"""全局配置：模型路径与运行参数。

所有路径按本机实际情况修改。下载指引见各模块顶部注释。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cfg:
    # ---------- ASR：sherpa-onnx 流式 Zipformer（中英双语）----------
    # 下载：https://github.com/k2-fsa/sherpa-onnx/releases（asr-models）
    # 推荐包：sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    asr_encoder: str = "models/asr/encoder.onnx"
    asr_decoder: str = "models/asr/decoder.onnx"
    asr_joiner: str = "models/asr/joiner.onnx"
    asr_tokens: str = "models/asr/tokens.txt"
    asr_bpe_vocab: str = "models/asr/bpe.model"  # 双语模型需要

    # ---------- LLM：MiniCPM5-1B GGUF（INT4，~0.5GB）----------
    # 下载：https://huggingface.co/openbmb/MiniCPM5-1B-GGUF
    # 推荐：MiniCPM5-1B-Q4_K_M.gguf
    llm_model: str = "models/llm/MiniCPM5-1B-Q4_K_M.gguf"
    llm_n_ctx: int = 8192          # 上下文窗口（MiniCPM5 原生 128K，按需缩）
    llm_n_gpu_layers: int = 0      # 0=纯CPU；有GPU时设大数(如 99)全部offload
    llm_n_threads: int = 4        # CPU 线程数
    llm_max_tokens: int = 256     # 单轮最大生成长度
    llm_temperature: float = 0.7

    # ---------- TTS：MOSS-TTS-Nano（语音克隆）----------
    # 参考音频：克隆目标音色（3–10 秒干净人声，48kHz/任意单声道）
    tts_ref_audio: str = "assets/voice_ref.wav"
    tts_voice: str = "zh_1"      # 无参考音频时的预设音色
    tts_sample_rate: int = 48000  # MOSS-TTS-Nano 输出 48kHz

    # ---------- 音频公共参数 ----------
    audio_sample_rate: int = 16000  # ASR 采样率
    audio_block: int = 1024         # 每次采样的帧数（~64ms @16kHz）
    audio_channels: int = 1

    # ---------- 端点检测（sherpa-onnx EndpointConfig 三条规则）----------
    # rule1：静音断句（trailing silence）。调小→更激进、延迟低、易误断。
    ep_min_trailing_silence: float = 0.8   # 秒；中文建议 0.6–0.9
    ep_min_utterance_length: float = 0.0   # rule1 触发：任一时刻静音达此值即断

    # ---------- Barge-in（打断）----------
    # 播放 TTS 时检测麦克风能量，超过阈值则判定用户插话。
    # ⚠️ 需要系统 AEC 回声消除，否则会被助手自己的声音误触发。
    bargein_enabled: bool = True
    bargein_energy_db: float = -35.0   # dBFS；-40 更灵敏，-30 更保守
    bargein_min_ms: int = 120          # 持续超阈值多久才算插话

    # ---------- 角色设定 ----------
    system_prompt: str = (
        "你是贾维斯，一位简洁、专业、低延迟的语音助手。"
        "用口语化短句回答，一次只说一到两句话，避免长篇大论和 Markdown。"
        "需要调用设备功能时用 XML 工具调用格式。"
    )


cfg = Cfg()
BASE_DIR = Path(__file__).resolve().parent
