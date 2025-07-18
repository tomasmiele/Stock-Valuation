from openai import OpenAI
from dotenv import load_dotenv
import os
import ast
import yfinance as yf
import statistics
import pandas as pd

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

tickers_info = {}

# Get the stock to be analysed
ticker = input("Type the stock ticker you want to evaluate: ").upper()

stock = yf.Ticker(ticker)

tickers_info[ticker] = {}

# Prompt and access ChatGPT to get similar companies to the analysed one
prompt = f"""List 4 US-listed companies similar to {ticker}, with valuation multiples available on yfinance. 
Output only a Python list of tickers."""

# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )

# content = response.choices[0].message.content
# tickers_list = ast.literal_eval(content)

tickers_list = ["PEP", "MNST", "KDP", "CELH"]

for t in tickers_list:
    tickers_info[t] = {}

# Calculate the trailing and forward PE of the stock
ticker_info = stock.info

trailing_PE = round(ticker_info.get("trailingPE"), 2)
forward_PE = round(ticker_info.get("forwardPE"), 2)

tickers_info[ticker]["Trailing PE"] = trailing_PE
tickers_info[ticker]["Forward PE"] = forward_PE

# Calculate the average trailing PE of the last 5y
ticker_price = stock.history(period="5y")["Close"].resample("YE").last()
ticker_price.index = ticker_price.index.tz_localize(None)

income = stock.income_stmt
net_income = income.loc["Net Income"].sort_index()
shares = income.loc["Basic Average Shares"].sort_index()

eps = (net_income / shares).rename("EPS")
eps.index = pd.to_datetime(eps.index)

price_df = ticker_price.reset_index().rename(columns={"Date": "date", "Close": "Price"})
eps_df = eps.reset_index().rename(columns={"index": "date", "EPS": "EPS"})

combined = pd.merge_asof(price_df.sort_values("date"), eps_df.sort_values("date"), on="date")
combined = combined[combined["date"].dt.year < pd.Timestamp.today().year]

combined["Trailing PE"] = combined["Price"] / combined["EPS"]
combined = combined.dropna()

ticker_five_y_avg_trailing_pe = float(round(combined["Trailing PE"].mean(), 2))

tickers_info[ticker]["5y Avg Trailing PE"] = ticker_five_y_avg_trailing_pe

# Repeat the process to the others companies
for t in tickers_list:
    peers =  yf.Ticker(t)
    peers_info = peers.info

    peers_trailing_PE = round(peers_info.get("trailingPE"), 2)
    peers_forward_PE = round(peers_info.get("forwardPE"), 2)

    tickers_info[t]["Trailing PE"] = peers_trailing_PE
    tickers_info[t]["Forward PE"] = peers_forward_PE

    peers_price = peers.history(period="5y")["Close"].resample("YE").last()
    peers_price.index = peers_price.index.tz_localize(None)

    peers_income = peers.income_stmt
    peers_net_income = peers_income.loc["Net Income"].sort_index()
    peers_shares = peers_income.loc["Basic Average Shares"].sort_index()

    eps = (peers_net_income / peers_shares).rename("EPS")
    eps.index = pd.to_datetime(eps.index)

    peers_price_df = peers_price.reset_index().rename(columns={"Date": "date", "Close": "Price"})
    peers_eps_df = eps.reset_index().rename(columns={"index": "date", "EPS": "EPS"})

    peers_combined = pd.merge_asof(peers_price_df.sort_values("date"), peers_eps_df.sort_values("date"), on="date")
    peers_combined = peers_combined[peers_combined["date"].dt.year < pd.Timestamp.today().year]

    peers_combined["Trailing PE"] = peers_combined["Price"] / peers_combined["EPS"]
    peers_combined = peers_combined.dropna()

    peers_five_y_avg_trailing_pe = float(round(peers_combined["Trailing PE"].mean(), 2))

    tickers_info[t]["5y Avg Trailing PE"] = peers_five_y_avg_trailing_pe

print(tickers_info)

trailing_pes = [v['Trailing PE'] for v in tickers_info.values()]
forward_pes = [v['Forward PE'] for v in tickers_info.values()]

avg_trailing_PE = round(statistics.mean(trailing_pes), 2)
median_trailing_PE = round(statistics.median(trailing_pes), 2)
avg_forward_PE = round(statistics.mean(forward_pes), 2)
median_forward_PE = round(statistics.median(forward_pes), 2)

cashflow = yf.Ticker(ticker).cashflow

#print(cashflow)