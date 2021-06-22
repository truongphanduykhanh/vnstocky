import os

from bs4 import BeautifulSoup
import requests

import credentials

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


export_url = 'https://www.cophieu68.vn/export.php'
login_url = 'https://www.cophieu68.vn/account/login.php'
payload = {
    'username': credentials.username,
    'tpassword': credentials.password,
    'ajax': 1,
    'login': 1
}

s = requests.Session()
s.post(login_url, data=payload, verify=False)

s_urls = s.get(export_url, verify=False).text
soup = BeautifulSoup(s_urls, 'lxml')
tds = soup.find_all('td', class_="td_bottom3 td_bg2", text='Download')


def get_download_url(td):
    url_id = td.a['href']
    url = f'https://www.cophieu68.vn/{url_id}'
    return url


# create new folders, if not existing
for table in ['events', 'excelfull', 'indexfinance', 'reportfinance']:
    data_path = os.path.join(os.getcwd(), 'data', table)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

# generate urls
urls = list(map(get_download_url, tds))
urls_events = [url for url in urls if 'events' in url]
urls_excelfull = [url for url in urls if 'excelfull' in url]
urls_indexfinance = [url for url in urls if 'indexfinance' in url]
urls_reportfinance = [url for url in urls if 'reportfinance' in url]


def download_file(url, encoding):
    ticker = url.split('/')[-1].split('=')[-1]
    table = url.split('/')[-1].split('.')[0]
    path = f'data/{table}/{ticker}_{table}.csv'
    r = s.get(url)
    content = r.content.decode(encoding)
    with open(path, 'w') as f:
        f.write(content)


list(map(lambda x: download_file(x, encoding='utf-16'), urls_events))
list(map(lambda x: download_file(x, encoding='utf-8'), urls_excelfull))
list(map(lambda x: download_file(x, encoding='utf-16'), urls_indexfinance))
list(map(lambda x: download_file(x, encoding='utf-16'), urls_reportfinance))
