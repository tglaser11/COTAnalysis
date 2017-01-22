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

'''
goldCOT_df columns:

Index([u'Open Interest', u'Producer/Merchant/Processor/User Longs',
       u'Producer/Merchant/Processor/User Shorts', u'Swap Dealer Longs',
       u'Swap Dealer Shorts', u'Swap Dealer Spreads', u'Money Manager Longs',
       u'Money Manager Shorts', u'Money Manager Spreads',
       u'Other Reportable Longs', u'Other Reportable Shorts',
       u'Other Reportable Spreads', u'Total Reportable Longs',
       u'Total Reportable Shorts', u'Non Reportable Longs',
       u'Non Reportable Shorts'],
      dtype='object')
'''

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

# Lookback Windows
lookback1 = 26
lookback2 = 52

'''
Open Interest Index
Normalized form of OI to determine if there is enthusiasm in the market.
High value may indicate over enthusiasm

OI Index = ((OI(recent) - OI(rollingmin)) / (OI(rollingmax) - OI(rollingmin))) * 100
'''

lb_window = goldCOT_df['Open Interest'].rolling(window = lookback1)
goldCOT_df['OIIndex1'] = ((goldCOT_df['Open Interest'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100

lb_window = goldCOT_df['Open Interest'].rolling(window = lookback2)
goldCOT_df['OIIndex2'] = ((goldCOT_df['Open Interest'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100


'''
Commercial Position Index
Normalized form of Commercial Position to determine what 'pros' are doing

CP = net commercial position
CP Index = ((CP(recent) - CP(rollingmin)) / (CP(rollingmax) - CP(rollingmin)) * 100
'''

goldCOT_df["netCP"] = goldCOT_df["Producer/Merchant/Processor/User Longs"] - goldCOT_df["Producer/Merchant/Processor/User Shorts"]

lb_window = goldCOT_df['netCP'].rolling(window = lookback1)
goldCOT_df['CPIndex1'] = ((goldCOT_df['netCP'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100

lb_window = goldCOT_df['netCP'].rolling(window = lookback2)
goldCOT_df['CPIndex2'] = ((goldCOT_df['netCP'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100


'''
Small Trader Position Index
Normalized form of Small Trader Position to determine what 'joe public' is doing

ST = net small trader position
ST Index = ((ST(recent) - ST(rollingmin)) / (ST(rollingmax) - ST(rollingmin)) * 100
'''

goldCOT_df["netST"] = goldCOT_df["Non Reportable Longs"] - goldCOT_df["Non Reportable Shorts"]

lb_window = goldCOT_df['netST'].rolling(window = lookback1)
goldCOT_df['STIndex1'] = ((goldCOT_df['netST'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100

lb_window = goldCOT_df['netST'].rolling(window = lookback2)
goldCOT_df['STIndex2'] = ((goldCOT_df['netST'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100



'''
Commercial Position as a Percentage of OI Index
Normalized form of Commercial Position relative to OI to determine what 'pros' are doing


CP% = CP Index / OI
CP% Index = ((CP%(recent) - CP%(rollingmin)) / (CP%(rollingmax) - CP%(rollingmin)) * 100
'''
goldCOT_df["CPPercent"] = goldCOT_df["netCP"] / goldCOT_df["Open Interest"]

lb_window = goldCOT_df['CPPercent'].rolling(window = lookback1)
goldCOT_df['CPPercentIndex1'] = ((goldCOT_df['CPPercent'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100

lb_window = goldCOT_df['netST'].rolling(window = lookback2)
goldCOT_df['CPPercentIndex2'] = ((goldCOT_df['CPPercent'] - lb_window.min()) / (lb_window.max() - lb_window.min())) * 100


goldCOT_df.tail()


