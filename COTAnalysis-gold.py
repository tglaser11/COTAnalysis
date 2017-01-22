'''
Gather Futures Commitment of Traders data and perform sentiment analysis using a methodology
similar to what Larry Williams uses in his book: Trade Stocks and Commodities with the Insiders.

Explore Gold, Silver and Copper

'''
import pandas as pd
import Quandl as quandl

# Gather Data
goldCOT_df = quandl.get('CFTC/GC_FO_ALL')
goldprice_df = quandl.get('CHRIS/CME_GC1', start_date="2006-06-01")


# Drop unwanted fields
goldprice_df = goldprice_df.drop(['Open','High','Low','Change','Settle','Volume','Open Interest'], axis=1)

# Combine datasets
goldCOT_df = pd.concat([goldCOT_df, goldprice_df], axis=1, join_axes=[goldCOT_df.index])
del goldprice_df

# Fix missing price data by inerpolating between data points using timeseries
goldCOT_df['Last'] = goldCOT_df['Last'].interpolate(method='time')


'''
Create a number of new feature types:
(1) Open Interest Index
(2) Commercial Position Index
(3) Small Trader Position Index
(4) Commerical / OI Index
(5) Trends: Price Up & OI Up (True / False)
(6) Trends: Price Up & OI Down (True / False)
(7) Trends: Price Down & OI Up (True / False)
(8) Trends: Price Down & OI Down (True / False)

'''

'''
Open Interest Index
Normalized form of OI to determine if there is enthusiasm in the market.
High value may indicate over enthusiasm

OI Index = ((OI(recent) - OI(min)) / (OI(max) - OI(min))) * 100


'''

# Open Interest Lookback Windows
OIlookback1 = 26
OILookback2 = 52

lb_window = goldCOT_df['Open Interest'].rolling(window = OIlookback1)
goldCOT_df['OIIndex1'] = ((goldCOT_df['Open Interest'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100

lb_window = goldCOT_df['Open Interest'].rolling(window = OIlookback2)
goldCOT_df['OIIndex2'] = ((goldCOT_df['Open Interest'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100





