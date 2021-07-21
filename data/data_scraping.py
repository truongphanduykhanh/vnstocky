import os

import requests
from bs4 import BeautifulSoup

import credentials

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning)


def create_session(credentials):
    '''
    Create and save a login session on the website

    Parameters:
    ----------
    credentials: a python script
        object 'username' and 'password' should exist in the python script

    Returns:
    -------
    a login session
    '''
    login_url = 'https://www.cophieu68.vn/account/login.php'
    payload = {
        'username': credentials.username,
        'tpassword': credentials.password,
        'ajax': 1,
        'login': 1
    }
    session = requests.Session()
    session.post(login_url, data=payload, verify=False)
    return session


def get_html_text(session):
    '''
    Get html in text format

    Parameters:
    ----------
    session:
        a login session on the website

    Returns:
    -------
    html in string format of the 'export' page
    '''
    export_url = 'https://www.cophieu68.vn/export.php'
    html_text = session.get(export_url, verify=False).text
    return html_text


def get_td_tags(html_text):
    '''
    Get a list of td tags in the input html text

    Parameters:
    ----------
    html_text: text of html

    Returns:
    -------
    list of tg tags
    '''
    soup = BeautifulSoup(html_text, 'lxml')
    tds = soup.find_all('td', class_="td_bottom3 td_bg2", text='Download')
    return tds


def get_file_url(td):
    '''
    Get downloadable url from a td tag

    Parameters:
    ----------
    td: html tg tag
        <td class="td_bottom3 td_bg2"
        align="center"><a href="export/events.php?id=A32"
        title="Download Lịch sự kiện" target="_blank">Download</a></td>

    Returns:
    -------
    csv file url
        Ex: https://www.cophieu68.vn/export/events.php?id=A32
    '''
    url_id = td.a['href']
    url = f'https://www.cophieu68.vn/{url_id}'
    return url


# create new four folders under data/, if not existing
for table in ['events', 'excelfull', 'indexfinance', 'reportfinance']:
    data_path = os.path.join(os.getcwd(), 'data', table)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)


# generate urls
session = create_session(credentials)
html_text = get_html_text(session)
tds = get_td_tags(html_text)
urls = list(map(get_file_url, tds))

urls_events = [url for url in urls if 'events' in url]
urls_excelfull = [url for url in urls if 'excelfull' in url]
urls_indexfinance = [url for url in urls if 'indexfinance' in url]
urls_reportfinance = [url for url in urls if 'reportfinance' in url]


def download_file(url, encoding):
    '''
    Download file from url

    Parameters:
    ----------
    url: csv url
        Ex: https://www.cophieu68.vn/export/events.php?id=A32
    encoding: encoding
        Ex: 'utf-16'

    Returns:
    -------
    save csv file to correct folder
        Ex: https://www.cophieu68.vn/export/events.php?id=A32
    '''
    ticker = url.split('/')[-1].split('=')[-1]
    table = url.split('/')[-1].split('.')[0]
    path = f'data/{table}/{ticker}_{table}.csv'
    r = session.get(url)
    content = r.content.decode(encoding)
    with open(path, 'w') as f:
        f.write(content)


list(map(lambda url: download_file(url, encoding='utf-16'), urls_events))
list(map(lambda url: download_file(url, encoding='utf-8'), urls_excelfull))
list(map(lambda url: download_file(url, encoding='utf-16'), urls_indexfinance))
list(map(lambda url: download_file(url, encoding='utf-16'), urls_reportfinance))
