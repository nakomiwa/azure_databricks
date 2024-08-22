# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. 必要なライブラリのインポート

# COMMAND ----------

import os
import base64
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. API設定
# MAGIC - APIキーは環境変数にクラスターの設定済み

# COMMAND ----------

# OpenAI APIキーを環境変数から取得
api_key = os.getenv("OPENAI_API_KEY")

# APIキーが正しく設定されているか確認
if not api_key:
    raise ValueError("OpenAI APIキーが設定されていません。環境変数を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 画像エンコード関数

# COMMAND ----------

def encode_image(image_path):
    """
    指定された画像ファイルをBase64エンコードする関数

    Args:
    image_path (str): 画像ファイルのパス

    Returns:
    str: Base64エンコードされた画像データ
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 免許証情報抽出関数
# MAGIC - SystemMessageで役割を設定し、HumanMessageで指示を出す。
# MAGIC - 英数字と記号は半角であること、苗字と名前の間には半角スペースを入れること等指示するとある程度抽出した情報を整形可能。

# COMMAND ----------

def extract_license_info(image_path):
    """
    運転免許証の画像から情報を抽出する関数（LangChainを使用）

    Args:
    image_path (str): 免許証画像のファイルパス

    Returns:
    dict: 抽出された情報の辞書
    """
    # 画像をBase64エンコード
    base64_image = encode_image(image_path)

    # ChatOpenAIモデルの初期化
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # プロンプトテンプレートの作成
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="あなたは画像から情報を抽出することができるAIアシスタントです。"),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "この画像は運転免許証です。名前、住所、電話番号（あれば）、免許証番号を抽出し、以下の形式で返してください。英数字と記号は半角でお願いします。日本人の場合は苗字と名前の間に半角スペースを入れてください。：\n名前: [抽出した名前]\n住所: [抽出した住所]\n電話番号: [抽出した電話番号]（なければ「記載なし」）\n免許証番号: [抽出した免許証番号]"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
    ])

    # プロンプトの実行
    response = chat(prompt.format_messages())

    # 結果の解析
    result = response.content
    info_dict = {}
    for line in result.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            info_dict[key.strip()] = value.strip()

    return info_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. フォルダ内の画像処理関数

# COMMAND ----------

def process_images_in_folder(folder_path):
    """
    指定されたフォルダ内のすべての画像を処理する関数

    Args:
    folder_path (str): 画像が格納されているフォルダのパス

    Returns:
    pd.DataFrame: 抽出された情報のDataFrame
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            info = extract_license_info(image_path)
            if info:
                info['ファイル名'] = filename
                results.append(info)
    
    return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. メイン処理
# MAGIC - ../data/image配下に複数の免許証の画像を格納。

# COMMAND ----------

if __name__ == "__main__":
    # 画像ファイルが格納されているフォルダのパスを相対パスで指定
    folder_path = "../data/image"
    
    # フォルダ内の画像を処理
    df = process_images_in_folder(folder_path)



# COMMAND ----------

#抽出された情報（DataFrame形式）
df
