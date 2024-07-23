# Databricks notebook source
# MAGIC %pip install openai python-docx

# COMMAND ----------

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from docx import Document
import os

# ChatGPTモデルの初期化
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

def generate_document():
    try:
        messages = [
            SystemMessage(content="あなたは日本語で技術マニュアルを作成する技術ライターです。SaaSのタスク管理ツール「CloudTaskPro」のマニュアルを作成してください。"),
            HumanMessage(content="CloudTaskProというSaaSタスク管理ツールの詳細なマニュアルを日本語で作成してください。ツールの機能、使用方法、ベストプラクティスを含めてください。アカウント管理、ダッシボードの使用法、タスクとプロジェクト管理、チーム協業、レポート作成、モバイルアプリの使用などのトピックをカバーしてください。約10,000文字（およそ5,000〜6,000字程度）を目安に作成してください。専門的で分かりやすい日本語を使用し、日本の企業文化に適した内容にしてください。")
        ]
        response = chat(messages)
        return response.content.strip()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def create_manual():
    content = generate_document()
    
    if content is None:
        print("ドキュメントの生成に失敗しました。")
        return

    # Word文書の作成
    document = Document()
    document.add_paragraph(content)

    document.save("CloudTaskPro_マニュアル.docx")
    print(f"マニュアルが生成されました。文字数: {len(content)}")

create_manual()
