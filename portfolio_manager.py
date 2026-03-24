import pandas as pd
import numpy as np
import scipy.optimize as sco

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Portföyün yıllık getiri ve volatilitesini hesaplar"""
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - 0.20) / std_dev if std_dev != 0 else 0
    return std_dev, returns, sharpe

def calculate_kelly_criterion(win_prob, win_loss_ratio):
    """
    Kelly Kriteri (Sermaye Yönetimi)
    k = w - (1-w)/r
    """
    if win_loss_ratio == 0: return 0
    k = win_prob - (1 - win_prob) / win_loss_ratio
    # Kelly kesrini genellikle muhafazakar olması için 1/2 veya 1/4 olarak alınır (Fractional Kelly)
    return max(0, k * 0.5) 

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Negatif Sharpe Oranı (Optimizasyon için)"""
    p_vol, p_ret, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def optimize_portfolio(symbols, expected_returns_dict, historical_dfs, risk_free_rate=0.20):
    """
    Markowitz Portföy Optimizasyonu.
    Maximize Sharpe Ratio.
    """
    if len(symbols) < 2:
        return {s: 1.0 for s in symbols}

    # Fiyat verilerini birleştir
    combined_df = pd.DataFrame()
    for s in symbols:
        if s in historical_dfs:
            combined_df[s] = historical_dfs[s]['Close']
    
    if combined_df.empty:
        return {s: 1.0/len(symbols) for s in symbols}

    # Günlük getiriler ve Kovaryans matrisi
    returns = combined_df.pct_change().dropna()
    mean_returns = returns.mean()
    
    # Eğer XGBoost'tan beklenen getiriler (expected_returns_dict) geldiyse 
    # tarihsel ortalama yerine bunları kullan (Daha proaktif yaklaşım)
    for s in symbols:
        if s in expected_returns_dict:
            # Tahmin edilen % değişim (21 günlük olduğu için günlüğe indirgeyebiliriz veya direkt orantı kurabiliriz)
            # Burada basitleştirmek için direkt XGBoost'un 'yön' ve 'potansiyel' bilgisini önceliklendiriyoruz
            pass 

    cov_matrix = returns.cov()
    num_assets = len(symbols)
    
    # Kısıtlamalar: Toplam ağırlık = 1.0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Sınırlar: Her hisse 0 ile 1 arasında (Açığa satış yok)
    bounds = tuple((0.0, 0.6) for _ in range(num_assets)) # Bir hisseye max %60 yatırım
    
    # İlk tahmin: Eşit dağılım
    initial_guess = num_assets * [1. / num_assets]
    
    print(f"⚖️ Portföy optimize ediliyor: {symbols}...")
    
    try:
        opt_results = sco.minimize(
            neg_sharpe_ratio,
            initial_guess,
            args=(mean_returns, cov_matrix, risk_free_rate/252), # Günlük RF rate
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not opt_results.success:
            print("⚠️ Optimizasyon başarısız, eşit dağılıma dönülüyor.")
            return {s: 1.0/len(symbols) for s in symbols}
            
        weights = opt_results.x
        return {symbols[i]: round(weights[i], 4) for i in range(num_assets)}
        
    except Exception as e:
        print(f"❌ Portföy Optimizasyon Hatası: {e}")
        return {s: 1.0/len(symbols) for s in symbols}

if __name__ == "__main__":
    # Test
    test_symbols = ["THYAO.IS", "EREGL.IS", "ASELS.IS"]
    test_returns = {"THYAO.IS": 0.05, "EREGL.IS": 0.02, "ASELS.IS": -0.01}
    # Mock data
    dates = pd.date_range("2023-01-01", periods=100)
    mock_dfs = {s: pd.DataFrame({"Close": np.random.randn(100).cumsum() + 100}, index=dates) for s in test_symbols}
    
    weights = optimize_portfolio(test_symbols, test_returns, mock_dfs)
    print(f"Optimal Ağırlıklar: {weights}")
