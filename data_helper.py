from googlesearch import search
import requests
from bs4 import BeautifulSoup
import yfinance as yf


def get_data(ticker_name):
    ticker = yf.Ticker(ticker_name+".NS")
    stock_info = ticker.info

    # 1. Get M&D
    url_md = ''
    for res in search(f"management discussion and analysis of {ticker_name} indiainfoline", num_results=1):
        url_md = res
        break

    r = requests.get(url_md)
    soup = BeautifulSoup(r.content, 'html.parser')

    s = soup.find('div', class_='Pyxilf')
    print('M&D')
    print('=================================')
    print(s.text)
    report_text = s.text

    content = f"""
        Management Discussion and Analysis:
        {report_text}
        
    """

    # 2. Key Ratios

    ratios = {
        'recommendation_key': stock_info['recommendationKey'],
        'industry': stock_info['industry'],

        'current_price': stock_info['currentPrice'],
        'db_to_eq': stock_info['debtToEquity'],
        'forward_pe': stock_info['forwardPE'],
        'pb': stock_info['priceToBook'],
        'roe': stock_info['returnOnEquity']
    }

    return content, ratios



