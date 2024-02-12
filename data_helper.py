from googlesearch import search
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nsepython import nsetools_get_quote


def get_data(ticker_name):
    # 1. Get M&D
    url_md = ''
    for res in search(f"management discussion and analysis of {ticker_name} indiainfoline", num_results=1):
        url_md = res
        break

    r = requests.get(url_md)
    soup = BeautifulSoup(r.content, 'html.parser')

    s = soup.find('div', class_='widget-content primary_contnent_bg p15')
    print('M&D')
    print('=================================')
    print(s.text)
    report_text = s.text

    # 2. Key financial ratios
    url_ratios = ''
    for res in search(f"{ticker_name} groww", num_results=1):
        url_ratios = res
        break

    # 2. Get Report content
    r = requests.get(url_ratios)
    soup = BeautifulSoup(r.content, 'html.parser')

    s = soup.find('table', class_='tb10Table col l12 ft785Table')
    df = pd.read_html(str(s))[0]

    price_data = nsetools_get_quote(ticker_name)

    # Extracting specific rows based on their indices
    roe = df.loc[1, 1]
    pe = df.loc[2, 1]
    eps = df.loc[3, 1]
    pb = df.loc[4, 1]
    div_yield = df.loc[5, 1]
    d_to_e = df.loc[8, 1]

    ratios = {'price': price_data['lastPrice'], 'roe': roe, 'pe': pe, 'eps': eps,
              'pb': pb, 'div_yield': div_yield, 'd_to_e': d_to_e}

    # 3. Get Cash Flow
    url_cashflow = ''
    for res in search(f"{ticker_name} consolidated cash flow moneycontrol", num_results=1):
        url_cashflow = res
        break

    r = requests.get(url_cashflow)
    soup = BeautifulSoup(r.content, 'html.parser')

    table_cashflow = soup.find('table', class_='mctable1')
    df_cashflow = pd.read_html(str(table_cashflow))[0]
    df_cashflow.columns = df_cashflow.iloc[0, 0:len(df_cashflow.columns)].values.astype(str)
    # Drop the first row as it's now redundant as column names
    df_cashflow = df_cashflow[2:]
    df_cashflow.dropna(inplace=True, thresh=2)
    df_cashflow.index = np.arange(1, len(df_cashflow) + 1)
    df_cashflow.pop(df_cashflow.columns[-1])

    # df
    print('Cash Flow')
    print('===================================')
    print(df_cashflow)

    # 4. Profit and Loss
    url_pl = ''
    for res in search(f"{ticker_name} consolidated profit and loss moneycontrol", num_results=1):
        url_pl = res
        break

    r = requests.get(url_pl)
    soup = BeautifulSoup(r.content, 'html.parser')

    table_pl = soup.find('table', class_='mctable1')
    df_pl = pd.read_html(str(table_pl))[0]
    df_pl.columns = df_pl.iloc[0, 0:len(df_pl.columns)].values.astype(str)
    # Drop the first row as it's now redundant as column names
    df_pl = df_pl[2:]
    df_pl.dropna(inplace=True, thresh=2)
    df_pl.index = np.arange(1, len(df_pl) + 1)
    df_pl.pop(df_pl.columns[-1])

    # df
    print('Profit and Loss')
    print('===================================')
    print(df_pl)

    # 5. Balance Sheet
    url_balance_sheet = ''
    for res in search(f"{ticker_name} consolidated balance sheet moneycontrol", num_results=1):
        url_balance_sheet = res
        break

    r = requests.get(url_balance_sheet)
    soup = BeautifulSoup(r.content, 'html.parser')

    table_bl = soup.find('table', class_='mctable1')
    df_bs = pd.read_html(str(table_bl))[0]
    df_bs.columns = df_bs.iloc[0, 0:len(df_bs.columns)].values.astype(str)
    # Drop the first row as it's now redundant as column names
    df_bs = df_bs[2:]
    df_bs.dropna(inplace=True, thresh=2)
    df_bs.index = np.arange(1, len(df_bs) + 1)
    df_bs.pop(df_bs.columns[-1])





    # df
    print('Balance Sheet')
    print('===================================')
    print(df_bs)

    content = f"""
        Management Discussion and Analysis:
        {report_text}
        
        Cash Flow Statement:
        {table_cashflow.contents}
        
        Profit and Loss Statement:
        {table_pl.contents}
        
        Balance Sheet Statement:
        {table_bl.contents}
    """
    return content, ratios, df_cashflow, df_pl, df_bs



