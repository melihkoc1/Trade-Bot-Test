"""Haber kaynağı fetcher modülleri."""

from news.fetchers.base_fetcher import BaseFetcher
from news.fetchers.investing_fetcher import InvestingFetcher
from news.fetchers.kap_fetcher import KAPFetcher
from news.fetchers.bigpara_fetcher import BigparaFetcher
from news.fetchers.foreks_fetcher import ForeksFetcher
from news.fetchers.tcmb_fetcher import TCMBFetcher

__all__ = [
    "BaseFetcher",
    "InvestingFetcher",
    "KAPFetcher",
    "BigparaFetcher",
    "ForeksFetcher",
    "TCMBFetcher",
]
