import yfinance as yf


df = yf.download("SPY", start="2010-01-01", end="2025-12-31", auto_adjust=True)
df.columns = df.columns.get_level_values(0)  # flatten the multi-header
df.to_csv("spy_data.csv")