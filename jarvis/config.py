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

    # ---------- VAD：Silero（sherpa-onnx VoiceActivityDetector）----------
    # 模型下载：https://github.com/k2-fsa/sherpa-onnx/releases（vad models）
    # 文件：silero_vad.onnx（int8 版 ~208KB）。放 models/vad/ 下。
    vad_model: str = "models/vad/silero_vad.onnx"
    vad_threshold: float = 0.6          # 语音概率阈值；0.5 灵敏，0.7 保守
    vad_min_silence_ms: int = 500       # 静音多久算一句话结束
    vad_speech_pad_ms: int = 200        # 语音段两端各补多少 ms（防止削头）

    # ---------- AEC：回声消除（可插拔后端，自动降级）----------
    # "webrtc" → 用 webrtc-audio-processing（需 pip install，效果最好）
    # "speex"  → 用 pyspeex 的 MDF AEC（备选）
    # "none"   → 不做软件 AEC，靠系统级 AEC 兜底（PulseAudio module-echo-cancel
    #            / macOS AVAudioEngine / Android AcousticEchoCanceler）
    # 运行时若所选后端不可用，自动降级到 "none" 并告警。
    aec_backend: str = "webrtc"
    aec_filter_length_ms: int = 200     # 自适应滤波器长度（speex/webrtc 建议 128–200ms）

    # ---------- Barge-in（打断，基于 VAD）----------
    # SPEAKING 状态下，若 VAD 在最近 window_ms 内检出任何语音段 → 判定插话。
    # 比 energy 阈值可靠：silero 能区分人声与噪声/残留回声。
    # 仍建议配合 AEC；纯 VAD 无 AEC 时，残留回声可能仍误触发（视环境而定）。
    bargein_enabled: bool = True
    bargein_window_ms: int = 300        # 回看时间窗
    bargein_min_speech_ms: int = 80     # 窗口内累计语音 ≥ 此值才触发

    # ---------- 角色设定 ----------
    system_prompt: str = (
        "你是贾维斯，一位简洁、专业、低延迟的语音助手。"
        "用口语化短句回答，一次只说一到两句话，避免长篇大论和 Markdown。"
        "需要调用设备功能时用 XML 工具调用格式。"
    )


cfg = Cfg()
BASE_DIR = Path(__file__).resolve().parent
