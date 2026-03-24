import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class CatalystManager:
    def __init__(self):
        self.cache = {}

    def get_catalysts(self, symbol):
        """
        Yahoo Finance üzerinden bilanço ve temettü tarihlerini çeker.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # 1. Bilanço Takvimi
            earnings_dates = ticker.calendar
            next_earnings = None
            if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
                # Calendar genellikle 'Earnings Date' satırı içeren bir tablo döner
                # Yfinance güncel sürümlerinde dict olarak da dönebilir
                if 'Earnings Date' in earnings_dates.index:
                    next_earnings = earnings_dates.loc['Earnings Date'].iloc[0]
            elif isinstance(earnings_dates, dict) and 'Earnings Date' in earnings_dates:
                next_earnings = earnings_dates['Earnings Date'][0]

            # 2. Temettü Bilgisi
            info = ticker.info
            ex_dividend_date = info.get("exDividendDate")
            dividend_yield = info.get("dividendYield", 0)
            
            if ex_dividend_date:
                ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime("%Y-%m-%d")

            # 3. Kalan Gün Hesabı
            days_to_earnings = None
            if next_earnings:
                if isinstance(next_earnings, str):
                    next_earnings_dt = datetime.strptime(next_earnings, "%Y-%m-%d")
                else: 
                    next_earnings_dt = next_earnings
                
                # Timezone varsa çıkaralım
                if hasattr(next_earnings_dt, 'tzinfo') and next_earnings_dt.tzinfo:
                    next_earnings_dt = next_earnings_dt.replace(tzinfo=None)
                    
                days_to_earnings = (next_earnings_dt - datetime.now()).days
                next_earnings = next_earnings_dt.strftime("%Y-%m-%d")

            return {
                "next_earnings": next_earnings,
                "days_to_earnings": days_to_earnings,
                "ex_dividend_date": ex_dividend_date,
                "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else 0
            }
        except Exception as e:
            print(f"Catalyst Tracking Error for {symbol}: {e}")
            return None

# Singleton
catalyst = CatalystManager()

def get_market_catalysts(symbol):
    return catalyst.get_catalysts(symbol)

if __name__ == "__main__":
    res = get_market_catalysts("THYAO.IS")
    print(f"THYAO Katalizörler: {res}")
