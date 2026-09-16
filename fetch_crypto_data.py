"""
=============================================================================
سيرفر جلب البيانات المجاني والموثوق - Free Reliable Data Fetcher
=============================================================================
المصدر: Binance Public API (منصة بايننس العالمية)
المميزات:
  - مجاني 100% وبدون أي API Key أو تسجيل دخول.
  - بيانات لحظية دقيقة (Real-Time Ticker).
  - بيانات تاريخية (Historical Candlesticks / OHLCV) مناسبة لتدريب النماذج والتحليل.
  - إمكانية حفظ البيانات تلقائياً في ملف CSV داخل مجلد data/.
=============================================================================
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any

# ضبط تشفير الطرفية على الويندوز لدعم UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import requests


class CryptoDataFetcher:
    """كلاس احترافي لاستدعاء البيانات من Binance Public API."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CryptoDataCollector/2.0 (Python-Client)"
        })

    def get_ticker_24h(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """
        جلب السعر اللحظي وملخص آخر 24 ساعة لرمز معين (مثل BTCUSDT, ETHUSDT, SOLUSDT).
        """
        symbol = symbol.upper()
        url = f"{self.BASE_URL}/ticker/24hr"
        params = {"symbol": symbol}

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            summary = {
                "symbol": data.get("symbol"),
                "last_price": float(data.get("lastPrice")),
                "price_change_percent": float(data.get("priceChangePercent")),
                "high_price_24h": float(data.get("highPrice")),
                "low_price_24h": float(data.get("lowPrice")),
                "volume": float(data.get("volume")),
                "quote_volume_usd": float(data.get("quoteVolume")),
                "timestamp": datetime.fromtimestamp(data.get("closeTime") / 1000).strftime("%Y-%m-%d %H:%M:%S")
            }
            return summary

        except requests.exceptions.RequestException as err:
            print(f"[ERROR] خطأ أثناء جلب سعر {symbol}: {err}")
            return None

    def get_historical_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        جلب الشموع التاريخية (OHLCV) وتحويلها إلى DataFrame جاهز للتحليل وتدريب النماذج.

        :param symbol: رمز العملة (مثل BTCUSDT)
        :param interval: الفريم الزمني ('1m', '5m', '15m', '1h', '4h', '1d')
        :param limit: عدد الشموع المطلوبة (الحد الأقصى 1000 شمعة)
        :return: pandas.DataFrame
        """
        symbol = symbol.upper()
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw_klines = response.json()

            columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]

            df = pd.DataFrame(raw_klines, columns=columns)

            # تحويل التواريخ
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            # تحويل القيم الرقمية إلى float
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
            for col in numeric_cols:
                df[col] = df[col].astype(float)

            # اختيار الأعمدة الهامة فقط
            df = df[["open_time", "open", "high", "low", "close", "volume", "number_of_trades"]]
            df.set_index("open_time", inplace=True)

            return df

        except requests.exceptions.RequestException as err:
            print(f"[ERROR] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {err}")
            return None

    def save_data(self, df: pd.DataFrame, filename: str = "crypto_historical.csv", folder: str = "data") -> str:
        """حفظ الـ DataFrame في ملف CSV."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath)
        print(f"[INFO] تم حفظ البيانات بنجاح في: {filepath}")
        return filepath


# =============================================================================
# التجربة التشغيلية (Run Demo)
# =============================================================================
if __name__ == "__main__":
    fetcher = CryptoDataFetcher()

    print("=" * 60)
    print(" 🚀 تجربة استدعاء البيانات من Binance Public API (مجاني 100%)")
    print("=" * 60)

    # 1. جلب الأسعار اللحظية لعدة عملات
    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    print("\n[1] الأسعار اللحظية لأشهر العملات:")
    print("-" * 60)
    for coin in coins:
        ticker = fetcher.get_ticker_24h(coin)
        if ticker:
            change_sign = "+" if ticker["price_change_percent"] >= 0 else ""
            print(f"🔸 {ticker['symbol']:<8} | السعر: ${ticker['last_price']:<10.2f} | التغير: {change_sign}{ticker['price_change_percent']:.2f}% | الحجم (24h): ${ticker['quote_volume_usd']:,.0f}")

    # 2. جلب الشموع التاريخية لـ Bitcoin وحفظها
    target_coin = "BTCUSDT"
    timeframe = "1h"
    candle_count = 24  # آخر 24 ساعة

    print(f"\n[2] جلب آخر {candle_count} شمعة تاريخية ({timeframe}) لـ {target_coin}:")
    print("-" * 60)
    history_df = fetcher.get_historical_data(symbol=target_coin, interval=timeframe, limit=candle_count)

    if history_df is not None:
        print(history_df.tail())
        print(f"\nإجمالي السجلات التي تم جلبها: {len(history_df)}")

        # حفظ البيانات في ملف CSV
        csv_path = fetcher.save_data(history_df, filename="btc_last_24h.csv")
        print(f"✅ تم الانتهاء بنجاح! مسار الملف: {csv_path}")
