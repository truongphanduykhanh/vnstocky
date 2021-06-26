class Params:
    ticker = None
    input_path = f'data/reportfinance/{ticker}_reportfinance.csv'
    start_bs = 'Tài sản ngắn hạn###Current Assets'
    end_bs = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'
    ticker_col = 'Ticker'
    time_col = 'Feat_Month'
    output_path = 'data/bs.csv'
    index = False
