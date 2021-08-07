'''
This script is to scrape raw data from cophieu68.vn.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-07-24'


import os

import requests
from bs4 import BeautifulSoup

import __credentials

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning)


class DataScraping:

    def __init__(self):
        self.session = None
        self.html_text = None
        self.tds = None
        self.urls = None

    def create_session(self, credentials):
        '''
        Create and save a login session on the website

        Parameters:
        ----------
        credentials: a python script
            Object 'username' and 'password' should exist in the python script

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
        self.session = session

    def get_html_text(self):
        '''
        Get html in text format

        Parameters:
        ----------
        session:
            A login session on the website

        Returns:
        -------
        html in string format of the 'export' page
        '''
        export_url = 'https://www.cophieu68.vn/export.php'
        html_text = self.session.get(export_url, verify=False).text
        self.html_text = html_text

    def get_td_tags(self):
        '''
        Get a list of td tags in the input html text

        Parameters:
        ----------
        html_text: text of html

        Returns:
        -------
        list of tg tags
        '''
        soup = BeautifulSoup(self.html_text, 'lxml')
        tds = soup.find_all('td', class_="td_bottom3 td_bg2", text='Download')
        self.tds = tds

    def get_file_urls(self):
        '''
        Get downloadable url from a td tag

        Parameters:
        ----------
        tds: list of html tg tag. Ex:
            [<td class="td_bottom3 td_bg2"
            align="center"><a href="export/events.php?id=A32"
            title="Download Lịch sự kiện" target="_blank">Download</a></td>]

        Returns:
        -------
        list of csv file url
            Ex: [https://www.cophieu68.vn/export/events.php?id=A32]
        '''
        url_ids = [td.a['href'] for td in self.tds]
        urls = [f'https://www.cophieu68.vn/{url_id}' for url_id in url_ids]
        self.urls = urls

    @staticmethod
    def create_folders(folders=[
        'events', 'excelfull', 'indexfinance', 'reportfinance'
    ]):
        # create new folders under current directory, if not existing
        for folder in folders:
            folder_path = os.path.join(os.getcwd(), folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

    def download_files(
        self,
        encodings={
            'events': 'utf-16',
            'excelfull': 'utf-8',
            'indexfinance': 'utf-16',
            'reportfinance': 'utf-16'
        }
    ):
        '''
        Download files from urls

        Parameters:
        ----------
        session:
            A login session on the website
        urls: list of csv urls
            Ex: [https://www.cophieu68.vn/export/events.php?id=A32]
        encodings: list of encodings
            Ex: ['utf-16']

        Returns:
        -------
        save csv files to correct folders
            Ex: https://www.cophieu68.vn/export/events.php?id=A32
        '''
        folders = [url.split('/')[-1].split('.')[0] for url in self.urls]
        tickers = [url.split('/')[-1].split('=')[-1] for url in self.urls]

        paths = []
        for folder, ticker in zip(folders, tickers):
            paths.append(f'{os.getcwd()}/{folder}/{ticker}_{folder}.csv')

        for url, path in zip(self.urls, paths):
            folder = url.split('/')[-1].split('.')[0]
            r = self.session.get(url)
            content = r.content.decode(encodings[folder])
            with open(path, 'w') as f:
                f.write(content)


if __name__ == '__main__':
    data_scraping = DataScraping()
    data_scraping.create_session(__credentials)
    data_scraping.get_html_text()
    data_scraping.get_td_tags()
    data_scraping.get_file_urls()
    DataScraping.create_folders()
    data_scraping.download_files()
