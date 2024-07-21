# Databricks notebook source
# MAGIC %md
# MAGIC # ライブラリインポート

# COMMAND ----------

import csv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# COMMAND ----------

# MAGIC %md
# MAGIC # 関数定義

# COMMAND ----------

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def get_customer_activities(data, customer_key, key_type):
    activities = []
    for row in data:
        if row[key_type] == customer_key:
            activities.append(f"訪問日時: {row['訪問日時']}\n活動内容: {row['活動内容']}")
    return activities

def generate_proposal(activities):
    llm = OpenAI(temperature=0.7)
    
    template = """
    あなたは大手銀行の優秀な法人営業マネージャーです。顧客との深い関係構築と、顧客のビジネスニーズに対する的確な提案が評価されています。以下は、重要顧客との過去の面談記録です：

    {activities}

    これらの面談記録を踏まえて、次回の訪問時に提案すべき事項の素案を3つ挙げてください。各提案は以下の要素を含めてください：

    1. 提案タイトル：簡潔で魅力的な提案名
    2. 背景：過去の面談記録から読み取れる顧客のニーズや課題
    3. 提案内容：具体的な製品、サービス、またはソリューション
    4. メリット：顧客にとってのメリットや期待される効果
    5. アプローチ方法：提案を効果的に伝えるための具体的な説明や資料
    6. フォローアップ計画：提案後の次のステップや継続的なサポート方法

    提案作成時の注意点：
    - 顧客の業界動向や市場環境を考慮に入れてください。
    - 当行の強みを活かした提案を心がけてください。
    - 長期的な関係構築を意識し、単なる商品販売ではなく、顧客のビジネス成長に寄与する提案を行ってください。
    - 必要に応じて、他部署（例：審査部、市場営業部、海外部など）との連携も提案に含めてください。
    - リスク管理の観点も忘れずに盛り込んでください。

    最後に、これらの提案をどのような順序で説明するのが最適か、その理由と共に簡潔に述べてください。

    それでは、プロフェッショナルな営業マネージャーとして、説得力のある提案を作成してください。
    """

    prompt = PromptTemplate(
        input_variables=["activities"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(activities="\n\n".join(activities))


# COMMAND ----------

# MAGIC %md
# MAGIC # メイン処理

# COMMAND ----------

# メイン処理
csv_file_path = './sales_contacts.csv'  # CSVファイルのパスを指定
data = read_csv(csv_file_path)
print(data)

# COMMAND ----------

# 特定の顧客を指定（例: 顧客番号で指定）
customer_key = "1234567"
key_type = "顧客番号"

activities = get_customer_activities(data, customer_key, key_type)

print(activities)

# COMMAND ----------

#提案素案を取得
if activities:
    proposal = generate_proposal(activities)
    print(f"顧客 {customer_key} に対する次回訪問時の提案素案：")
    print(proposal)
else:
    print(f"顧客 {customer_key} の活動記録が見つかりません。")
