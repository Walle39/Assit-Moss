"""入口：python run.py

启动前请按 config.py 配好三个模型路径，并确认系统已装 PortAudio。
"""

from __future__ import annotations

import signal
import sys

from orchestrator import Orchestrator


def main() -> int:
    orch = Orchestrator()
    try:
        orch.start()
        signal.pause()  # 主线程挂起，等 Ctrl+C
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"[Jarvis] 启动失败: {e}")
        return 1
    finally:
        orch.stop()
        print("[Jarvis] 已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
