"""
SBI証券API連携モジュール

SBI証券へのログインおよび売買操作を提供する
戻り値はdict形式とし、HTTPレスポンス形式への変換はAPI層で行う
"""

from typing import Tuple, Dict, Any


def sbi_login(user_id: str, password: str) -> Tuple[Dict[str, Any], int]:
    """
    SBI証券へログインします。
    
    Args:
        user_id: ログインID
        password: パスワード
    
    Returns:
        (結果dict, HTTPステータスコード) のタプル
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        # ChromeOptionsの設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # ヘッドレスモードで実行

        # WebDriverの初期化 (Chromeをheadlessモードで起動)
        driver = webdriver.Chrome(options=chrome_options)

        # SBI証券のログインページにアクセス
        driver.get("https://www.sbisec.co.jp/")

        # ログインIDとパスワードを入力
        login_id_element = driver.find_element(By.ID, "login_id")
        login_password_element = driver.find_element(By.ID, "login_password")
        login_id_element.send_keys(user_id)
        login_password_element.send_keys(password)

        # ログインボタンをクリック
        login_button = driver.find_element(By.ID, "login_button")
        login_button.click()

        # ログイン後の処理 (例: ログイン成功の確認)
        try:
            # ログイン後のページに特定の要素が存在するか確認する (例: 口座情報)
            # ログイン後のページに特定の要素が存在するか確認する (例: 口座情報)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "user_name"))  # ユーザー名のIDを仮定
            )
            login_status = "ok"
            # 口座情報を取得する処理を実装する (例: ユーザー名を取得)
            account_info_element = driver.find_element(By.ID, "user_name") # ユーザー名のIDを仮定
            account_info = account_info_element.text if account_info_element else "口座情報取得失敗"
        except:
            login_status = "failed"
            account_info = None

        # WebDriverを終了
        driver.quit()

        if login_status == "ok":
            return {
                "status": "ok",
                "message": "SBI login successful.",
                "account_info": account_info
            }, 200
        else:
            return {
                "status": "failed",
                "message": "SBI login failed."
            }, 401

    except Exception as e:
        return {
            "error": str(type(e).__name__),
            "message": str(e)
        }, 500

def sbi_trade(user_id: str, password: str, symbol: str, quantity: int, trade_type: str) -> Tuple[Dict[str, Any], int]:
    """
    SBI証券で株式を売買します。
    
    Args:
        user_id: ログインID
        password: パスワード
        symbol: 銘柄コード
        quantity: 数量
        trade_type: 売買区分 ("buy" or "sell")
    
    Returns:
        (結果dict, HTTPステータスコード) のタプル
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        # ChromeOptionsの設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # ヘッドレスモードで実行

        # WebDriverの初期化 (Chromeをheadlessモードで起動)
        driver = webdriver.Chrome(options=chrome_options)

        # SBI証券のログインページにアクセス
        driver.get("https://www.sbisec.co.jp/")

        # ログインIDとパスワードを入力
        login_id_element = driver.find_element(By.ID, "login_id")
        login_password_element = driver.find_element(By.ID, "login_password")
        login_id_element.send_keys(user_id)
        login_password_element.send_keys(password)

        # ログインボタンをクリック
        login_button = driver.find_element(By.ID, "login_button")
        login_button.click()

        # ログイン後の処理 (例: ログイン成功の確認)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "user_name"))  # ユーザー名のIDを仮定
        )

        # ここに株式売買のダミーフローを実装
        # 例: 銘柄検索、注文画面への遷移、数量入力、売買区分選択、注文実行など
        try:
            # 銘柄検索（仮実装）
            # driver.find_element(By.ID, "search_box").send_keys(symbol)
            # driver.find_element(By.ID, "search_button").click()

            # 注文画面への遷移（仮実装）
            # driver.find_element(By.ID, "order_button").click()

            # 数量入力・売買区分選択（仮実装）
            # driver.find_element(By.ID, "quantity").send_keys(str(quantity))
            # driver.find_element(By.ID, trade_type).click()  # trade_type: "buy" or "sell"

            # 注文実行（仮実装）
            # driver.find_element(By.ID, "submit_order").click()

            trade_status = "success"
            trade_message = f"{symbol}を{quantity}株{trade_type}注文（ダミー）完了"
        except Exception as e:
            trade_status = "failed"
            trade_message = f"注文処理中にエラー: {str(e)}"

        # WebDriverを終了
        driver.quit()

        return {
            "status": trade_status,
            "message": trade_message,
            "trade_status": trade_status
        }, 200

    except Exception as e:
        return {
            "error": str(type(e).__name__),
            "message": str(e)
        }, 500
