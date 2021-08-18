# Vnstocky

Vnstocky is a Python library for dealing with Vietnamese stock prices prediction.

## Installation

Use user interface of GitHub or git in terminal to download the repository:

```bash
git clone https://github.com/truongphanduykhanh/vnstocky.git
```

## Usage

1. Create file *__credentials.py*  in folder */data*. This file should have variables *username* and *password*, which are credentials to log in [cophieu68.vn](http://cophieu68.vn).

2. Execute the script *data_scraping.py* to scrap data from [cophieu68.vn](http://cophieu68.vn):
```
python data_scraping.py
```

3. Execute the script *label_gen.py* to generate labels which are relative returns of stocks:
```
python label_gen.py
```

4. Execute the script *feature_gen.py* to generate features which include income statement, balance sheet and their various derivatives:
```
python feature_gen.py
```

5. Execute the script *model_gen.py* to generate list of tickers which are the most worthiest to invest:
```
python model_gen.py
```


## License
The repository are free for individual use.