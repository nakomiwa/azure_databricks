# Databricks notebook source
# MAGIC %md
# MAGIC # 営業支援プログラム
# MAGIC
# MAGIC このノートブックでは、若手営業職員向けの営業支援プログラムを実装します。
# MAGIC 営業活動データとレコメンド情報を活用し、OpenAI APIを利用して提案内容の素案を生成します。

# COMMAND ----------

# 必要なライブラリのインポート

import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# COMMAND ----------

# MAGIC %md
# MAGIC # データの読み込みと確認
# MAGIC
# MAGIC 営業活動データ（sales_contact.csv）とレコメンド情報（recommend.csv）を読み込み、内容を確認します。

# COMMAND ----------

# 営業活動データの読み込みと表示
sales_contact_df = pd.read_csv('./sales_contacts.csv')
print("営業活動データ:")
print(sales_contact_df)
print("\n")

# レコメンド情報の読み込みと表示
recommend_df = pd.read_csv('./recommend.csv')
print("レコメンド情報:")
print(recommend_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 対象企業の営業活動データの取得
# MAGIC
# MAGIC 指定された店番と顧客番号に基づいて、対象企業の営業活動データを取得します。

# COMMAND ----------

# 7桁の顧客番号の指定
customer_number = '1234567'

# 対象企業の営業活動データを取得（訪問日時と活動内容のみ）
target_company_data = sales_contact_df[
    sales_contact_df['顧客番号'].astype(str).str.zfill(7) == customer_number
][['訪問日時', '活動内容']]

# データが見つからない場合のエラーハンドリング
if target_company_data.empty:
    print(f"顧客番号 {customer_number} に該当する企業が見つかりません。")
else:
    print("対象企業の営業活動データ（訪問日時と活動内容）:")
    print(target_company_data)

# 企業名の取得（後続の処理で使用）
company_name = sales_contact_df[
    sales_contact_df['顧客番号'].astype(str).str.zfill(7) == customer_number
]['顧客名'].values[0] if not target_company_data.empty else "不明"

# 営業活動データを文字列に変換（後続の処理で使用）
sales_data = target_company_data.to_string(index=False) if not target_company_data.empty else "データなし"

# COMMAND ----------

# MAGIC %md
# MAGIC # レコメンド情報の取得
# MAGIC
# MAGIC レコメンド情報の1件目を取得します。

# COMMAND ----------

# レコメンド情報の取得（1件目のみ）
recommend_info = recommend_df.to_string

# COMMAND ----------

recommend_info

# COMMAND ----------

# MAGIC %md
# MAGIC # プロンプトの組み立て
# MAGIC
# MAGIC LangchainのPromptTemplateを使用してプロンプトを組み立てます。

# COMMAND ----------

# PromptTemplateの定義
prompt_template = PromptTemplate(
    input_variables=["company_name", "sales_data", "recommend_info"],
    template="""
あなたは経験豊富な営業マネージャーです。若手営業職員に対して、効果的な営業提案の指導を行ってください。

対象企業: {company_name}

これまでの営業活動データ:
{sales_data}

レコメンド情報:
{recommend_info}

上記の情報を踏まえて、以下の点を考慮した営業提案の素案を作成してください：

1. 顧客のニーズと課題を分析し、それに沿った提案を行う
2. これまでの営業活動の内容を活かし、顧客との関係性を深める提案をする
3. レコメンド情報を効果的に活用し、具体的な商品やサービスを提案する
4. 若手営業職員でも実行可能な、具体的かつ実践的な提案内容にする
5. 提案のセールスポイントを明確に示す

提案の構成：
1. はじめに：顧客の現状理解と課題の確認
2. 提案内容：具体的な解決策と商品・サービスの提案
3. 期待される効果：提案導入後のメリットの説明
4. 次のステップ：提案実現に向けたアクションプラン

また、若手営業職員向けのアドバイスも付け加えてください。
"""
)

# プロンプトの組み立て
prompt = prompt_template.format(
    company_name=company_name,
    sales_data=sales_data,
    recommend_info=recommend_info
)

print("組み立てたプロンプト:")
print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC # OpenAI APIへの連携と回答の取得
# MAGIC
# MAGIC 組み立てたプロンプトをOpenAI APIに送信し、回答を取得します。

# COMMAND ----------

# OpenAI モデルの設定
llm = OpenAI(
    temperature=0,
    max_tokens=1500
)

# LLMChainの作成と実行
chain = LLMChain(llm=llm, prompt=prompt_template)
response = chain.run({
    "company_name": company_name,
    "sales_data": sales_data,
    "recommend_info": recommend_info
})

print("AI生成の営業提案素案:")
print(response)

# COMMAND ----------


