from typing import Dict, List


class CapabilityEngine:
    """
    AIデベロッパーの能力を定義し、実行するエンジンクラス。
    """

    def __init__(self) -> None:
        """
        エンジンの初期化。
        """
        self.capabilities: List[Dict[str, str]] = [
            {"name": "Code Generation", "desc": "高品質なPythonコードの生成"},
            {"name": "Project Scaffolding", "desc": "ディレクトリ構造とファイル構成の自動構築"},
            {"name": "Refactoring", "desc": "既存コードの最適化とクリーンアップ"},
            {"name": "Automation", "desc": "PowerShellやPythonによる業務自動化"},
            {"name": "Debugging", "desc": "エラー解析とバグ修正案の提示"},
        ]

    def get_all_capabilities(self) -> List[Dict[str, str]]:
        """
        登録されているすべての機能リストを返します。

        Returns:
            List[Dict[str, str]]: 機能名と説明のリスト
        """
        return self.capabilities

    def run_demo(self, formatter) -> None:
        """
        デモを実行し、機能を一覧表示します。

        Args:
            formatter: TextFormatter インスタンス (またはそれに準ずるオブジェクト)
        """
        formatter.print_header("AI Developer Capabilities Demo")

        features = self.get_all_capabilities()
        for feature in features:
            formatter.print_feature(feature["name"], feature["desc"])

        print("\n[!] Status: All systems operational.")
