from openai import OpenAI
from dotenv import load_dotenv
import os
import ast

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

ticker = input("Type the stock ticker you want to evaluate: ").upper()

prompt = f"""List 4 US-listed companies similar to {ticker}. Output only a Python list of tickers."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

content = response.choices[0].message.content
tickers_list = ast.literal_eval(content)

print(tickers_list)