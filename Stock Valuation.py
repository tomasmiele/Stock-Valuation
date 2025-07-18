from openai import OpenAI
from dotenv import load_dotenv
import os
import ast
import yfinance as yf
import statistics
import pandas as pd
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

tickers_info = {}

# Get the stock to be analysed
ticker = input("Type the stock you want to evaluate (e.g. 'KO'): ").upper()

stock = yf.Ticker(ticker)

ticker_info = stock.info

industry = ticker_info["industry"]

tickers_info[ticker] = {}

# Prompt and access ChatGPT to get similar companies to the analysed one
prompt = f"""prompt = f'List 4 US-listed companies in the "{industry}" industry, similar to {ticker}, 
with valuation multiples available on yfinance. Output only a Python list of tickers."""

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )

# content = response.choices[0].message.content
# tickers_list = ast.literal_eval(content)

tickers_list = ["PEP", "KDP", "MNST", "CCEP"]

for t in tickers_list:
    tickers_info[t] = {}

# 1st Method - PE Multiples

# Calculate the trailing and forward PE of the stock
trailing_PE = round(ticker_info.get("trailingPE"), 2)
forward_PE = round(ticker_info.get("forwardPE"), 2)

tickers_info[ticker]["Multiple"] = {}

tickers_info[ticker]["Multiple"]["Trailing PE"] = trailing_PE
tickers_info[ticker]["Multiple"]["Forward PE"] = forward_PE

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

tickers_info[ticker]["Multiple"]["5y Avg Trailing PE"] = ticker_five_y_avg_trailing_pe

# Repeat the process to the others companies
for t in tickers_list:
    peers =  yf.Ticker(t)
    peers_info = peers.info

    peers_trailing_PE = round(peers_info.get("trailingPE"), 2)
    peers_forward_PE = round(peers_info.get("forwardPE"), 2)

    tickers_info[t]["Multiple"] = {}

    tickers_info[t]["Multiple"]["Trailing PE"] = peers_trailing_PE
    tickers_info[t]["Multiple"]["Forward PE"] = peers_forward_PE

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

    tickers_info[t]["Multiple"]["5y Avg Trailing PE"] = peers_five_y_avg_trailing_pe

# Calculate the average and mean PE's from the 5 companies combined
trailing_pes = [v["Multiple"]['Trailing PE'] for v in tickers_info.values()]
forward_pes = [v["Multiple"]['Forward PE'] for v in tickers_info.values()]

avg_trailing_PE = round(statistics.mean(trailing_pes), 2)
median_trailing_PE = round(statistics.median(trailing_pes), 2)
avg_forward_PE = round(statistics.mean(forward_pes), 2)
median_forward_PE = round(statistics.median(forward_pes), 2)

# Calculate the difference between the average and median for each company
tickers_info[ticker]["Multiple"]["PE Diff Indust Avg"] = round(avg_trailing_PE / tickers_info[ticker]["Multiple"]["Trailing PE"] * 100 - 100, 2)
tickers_info[ticker]["Multiple"]["PE Diff Indust Median"] = round(median_trailing_PE / tickers_info[ticker]["Multiple"]["Trailing PE"] * 100 - 100, 2)
tickers_info[ticker]["Multiple"]["Forward PE Diff Indust Avg"] = round(avg_forward_PE / tickers_info[ticker]["Multiple"]["Forward PE"] * 100 - 100, 2)
tickers_info[ticker]["Multiple"]["Forward PE Diff Indust Median"] = round(median_forward_PE / tickers_info[ticker]["Multiple"]["Forward PE"] * 100 - 100, 2)
tickers_info[ticker]["Multiple"]["5y PE Diff"] = round(tickers_info[ticker]["Multiple"]["5y Avg Trailing PE"] / tickers_info[ticker]["Multiple"]["Trailing PE"] * 100 - 100, 2)

for t in tickers_list:
    tickers_info[t]["Multiple"]["PE Diff Indust Avg"] = round(avg_trailing_PE / tickers_info[t]["Multiple"]["Trailing PE"] * 100 - 100, 2)
    tickers_info[t]["Multiple"]["PE Diff Indust Median"] = round(median_trailing_PE / tickers_info[t]["Multiple"]["Trailing PE"] * 100 - 100, 2)
    tickers_info[t]["Multiple"]["Forward PE Diff Indust Avg"] = round(avg_forward_PE / tickers_info[t]["Multiple"]["Forward PE"] * 100 - 100, 2)
    tickers_info[t]["Multiple"]["Forward PE Diff Indust Median"] = round(median_forward_PE / tickers_info[t]["Multiple"]["Forward PE"] * 100 - 100, 2)
    tickers_info[t]["Multiple"]["5y PE Diff"] = round(tickers_info[t]["Multiple"]["5y Avg Trailing PE"] / tickers_info[t]["Multiple"]["Trailing PE"] * 100 - 100, 2)

# 2nd Method - DCF

# Calculate the cashflow
tickers_info[ticker]["DCF"] = {}

cashflow = stock.cashflow

try:
    last_4_years = cashflow.columns[:4]
except Exception as e:
    print(f"[{ticker}] Erro ao acessar colunas do cashflow: {e}")

if "Free Cash Flow" in cashflow.index:
    fcf = cashflow.loc["Free Cash Flow", last_4_years]
else:
    try:
        ocf = cashflow.loc["Operating Cash Flow", last_4_years]
        capex = cashflow.loc["Capital Expenditures", last_4_years]
        fcf = ocf - capex
    except KeyError as e:
        print(f"[{ticker}] Dados insuficientes para calcular FCF: {e}")

fcf = fcf.sort_index()

tickers_info[ticker]["DCF"]["FCF"] = fcf

y0 = fcf.iloc[-3]
y1 = fcf.iloc[-2]
y2 = fcf.iloc[-1]

drop_ratio = (y1 - y2) / y1

if drop_ratio > 0.3:
    prev_growth = y1 / y0
    adjusted_y2 = y1 * prev_growth
    fcf.iloc[-1] = adjusted_y2

tickers_info[ticker]["DCF"]["Ajusted FCF"] = fcf

# Calculate the growth
fcf_growth = round(fcf.sort_index().pct_change().dropna() * 100, 2)
historical_growth = fcf_growth.mean()

historical_growth = 0.035

cagr_prompt = f"""Search for the forecasted CAGR of the "{industry}" industry **specifically** 
from either Grand View Research (GVR) or Statista. If available, return **only** the CAGR number. 
Do not include any units, percent signs, or explanation."""

# cagr_response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": cagr_prompt}
#     ]
# )

# cagr = cagr_response.choices[0].message.content.strip()

# match = re.search(r"[\d.]+", cagr)
# if match:
#     cagr = float(match.group())

cagr = 0.064

growth = 0.8 * historical_growth + 0.2 * cagr

tickers_info[ticker]["DCF"]["Historical Growth"] = historical_growth
tickers_info[ticker]["DCF"]["CAGR"] = cagr
tickers_info[ticker]["DCF"]["Estimate Growth"] = growth

# Forecast
projected_fcf = []
last_fcf = fcf.iloc[-1]

for i in range(1, 11):
    projected_value = last_fcf * ((1 + growth / 100) ** i)
    projected_fcf.append(float(round(projected_value, 2)))

tickers_info[ticker]["DCF"]["Projected FCF"] = projected_fcf

ticker_PFCF = ticker_info["marketCap"] / fcf
ticker_PFCF = float(round(ticker_PFCF.median(), 2))

tickers_info[ticker]["DCF"]["PFCF"] = ticker_PFCF
tickers_info[ticker]["DCF"]["TV - PFCF"] = projected_fcf[-1] * ticker_PFCF

tnx = yf.Ticker("^TNX")

r = (round(tnx.history(period="1d")["Close"].iloc[-1] , 2) + 3) / 100

tickers_info[ticker]["DCF"]["TV - Perpetuity"] = float(projected_fcf[-1] * (1 + growth) / (r - growth))

discounted_fcf = []
sum_dis_fcf = 0

for i in range(10):
    year = i + 1  
    discounted_value = projected_fcf[i] / ((1 + r) ** year)
    discounted_fcf.append(float(round(discounted_value, 2)))
    sum_dis_fcf += discounted_value

tickers_info[ticker]["DCF"]["Discounted FCF"] = discounted_fcf
tickers_info[ticker]["DCF"]["Discounted TV - PFCF"] = float(tickers_info[ticker]["DCF"]["TV - PFCF"] / ((1 + r) ** 10))
tickers_info[ticker]["DCF"]["Discounted TV - Perpetuity"] = float(tickers_info[ticker]["DCF"]["TV - Perpetuity"] / ((1 + r) ** 10) )
tickers_info[ticker]["DCF"]["Valuation - PFCF"] = float(discounted_value + tickers_info[ticker]["DCF"]["Discounted TV - PFCF"])
tickers_info[ticker]["DCF"]["Valuation - Perpetuity"] = float(discounted_value + tickers_info[ticker]["DCF"]["Discounted TV - Perpetuity"])

# Repeat the process to the others companies

print(tickers_info)