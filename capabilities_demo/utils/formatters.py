class TextFormatter:
    """
    テキストの出力フォーマットを管理するユーティリティクラス。
    """

    @staticmethod
    def print_header(title: str) -> None:
        """
        見出しを装飾して表示します。

        Args:
            title (str): 表示するタイトル文字列
        """
        border = "=" * (len(title) + 4)
        print(f"\n{border}")
        print(f"  {title}  ")
        print(f"{border}")

    @staticmethod
    def print_feature(feature_name: str, description: str) -> None:
        """
        機能とその説明を整形して表示します。

        Args:
            feature_name (str): 機能の名前
            description (str): 機能の説明
        """
        print(f"[*] {feature_name.ljust(20)} : {description}")
