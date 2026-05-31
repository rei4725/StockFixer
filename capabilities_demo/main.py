import sys

from core.engine import CapabilityEngine

from utils.formatters import TextFormatter


def main() -> None:
    """
    デモプログラムのエントリーポイント。
    """
    # フォーマッタの初期化
    formatter = TextFormatter()

    # エンジンの初期化
    engine = CapabilityEngine()

    try:
        # デモ実行
        engine.run_demo(formatter)
    except Exception as e:
        print(f"An error occurred during demo execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
