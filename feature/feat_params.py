class Params:
    ticker = None
    input_path = None
    start_bs = 'Tài sản ngắn hạn###Current Assets'
    end_bs = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'
    ticker_col = 'Ticker'
    time_col = 'Feat_Month'
    output_path = 'data/bs.csv'
    index = False

    def gen_input_path(ticker):
        return f'data/reportfinance/{ticker}_reportfinance.csv'
