import pandas as pd

from feat_gen import Finance
from feat_params import Params


tickers = ['TCB', 'TN1', 'VCB']

bs_df = []
for i, ticker in enumerate(tickers):
    print(f'{i:5}/{len(tickers)} ---> Processing {ticker}...')
    Params.ticker = ticker
    Params.input_path = Params.input_path.replace('None', ticker)
    finance = Finance()
    finance.load_data(**Params.__dict__)
    finance.get_bs(**Params.__dict__)
    finance.get_bs_qtr(**Params.__dict__)
    bs_df.append(finance.bs_qtr)

print(f' ----------> Done')
bs_df = pd.concat(bs_df).reset_index(drop=True)
bs_df.to_csv(Params.__dict__['output_path'], index=False)
