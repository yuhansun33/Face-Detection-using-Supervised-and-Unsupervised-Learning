import requests
from bs4 import BeautifulSoup
import os

# 新聞網的URL
url = 'https://www.msn.com/zh-tw/channel/topic/%E7%84%A6%E9%BB%9E%E6%96%B0%E8%81%9E/tp-Y_321db332-7d5a-4672-96b5-2d342ca554fb'

# 設定圖片保存的文件夾
image_dir = 'images'
os.makedirs(image_dir, exist_ok=True)

# 使用requests獲取網頁內容
response = requests.get(url)
web_content = response.text

# 使用BeautifulSoup解析網頁
soup = BeautifulSoup(web_content, 'html.parser')

# 尋找網頁中所有的圖片標籤
images = soup.find_all('img')

# 下載每張圖片
for img in images:
    # 獲取圖片的URL
    img_url = img.get('src')
    
    # 如果img_url是None，跳過此次循環
    if img_url is None:
        continue
    
    # 跳過不包含標準圖片格式的圖片URL
    if not any(ext in img_url for ext in ['.jpg', '.jpeg', '.png']):
        continue
    
    # 檢查URL是否完整
    if not img_url.startswith(('http:', 'https:')):
        img_url = 'https:' + img_url
    
    # 獲取圖片的文件名
    filename = os.path.join(image_dir, img_url.split('/')[-1].split('?')[0]) # 移除URL的參數
    
    try:
        # 獲取圖片內容
        img_data = requests.get(img_url).content
        # 保存圖片
        with open(filename, 'wb') as handler:
            handler.write(img_data)
            print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Could not download {img_url}. Reason: {e}")
