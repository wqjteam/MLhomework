from urllib import request
from urllib.request import urlretrieve
from urllib.parse import urlencode
import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ActionChains
import pyautogui
import time

url = 'https://baike.baidu.com/item/'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36'
headers = {'User-Agent': user_agent}


def crawlBDBK(item_name):
    new_url = url + urllib.parse.quote(item_name)
    driver = webdriver.Chrome(
        executable_path='C:\Program Files (x86)\Google\Chrome\Application\plugins\chromedriver_win32\chromedriver.exe')
    driver.get(new_url)
    print('请求的拼接url:%s' % {new_url})
    # 解析网页
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    description = soup.find(attrs={"name": "description"})['content']
    action = ActionChains(driver)
    # 获取图片
    simple_src = driver.find_elements_by_class_name('summary-pic')[0] \
        .find_element_by_tag_name('img').get_attribute('src')
    summary_img_src = driver.find_elements_by_class_name('summary-pic')[0] \
        .find_element_by_tag_name('a').get_attribute(
        'href')
    req = request.Request(summary_img_src, headers={
        'User-Agent': r'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0'})
    urlretrieve(url=simple_src, filename='./../../download/' + item_name + '-3.jpg')

    print("")
    # button.click()
    # print(imgs)
    # print(soup.prettify())


if __name__ == '__main__':
    crawlBDBK('折线图')
