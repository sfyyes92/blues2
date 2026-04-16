import datetime
import requests
from bs4 import BeautifulSoup
import os

# 1. 自动生成当天的URL
today = datetime.datetime.now().day
base_url = "https://us1.zhuk.dpdns.org/"
page_url = f"{base_url}{today}-Youtube.html"

print(f"✅ 自动生成今日链接: {page_url}")

# 2. 爬取页面，提取c.yaml和v.txt的链接
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    response = requests.get(page_url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # 3. 提取c.yaml和v.txt的完整链接
    c_yaml_url = None
    v_txt_url = None

    # 查找所有a标签，匹配文件名
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "c.yaml" in href:
            # 补全完整URL
            if not href.startswith("http"):
                c_yaml_url = base_url + href.lstrip("/")
            else:
                c_yaml_url = href
        elif "v.txt" in href:
            if not href.startswith("http"):
                v_txt_url = base_url + href.lstrip("/")
            else:
                v_txt_url = href

    if not c_yaml_url or not v_txt_url:
        print("❌ 未找到c.yaml或v.txt文件，请检查页面是否正常加载")
        exit(1)

    print(f"✅ 找到c.yaml: {c_yaml_url}")
    print(f"✅ 找到v.txt: {v_txt_url}")

    # 4. 自动下载文件
    os.makedirs("node_files", exist_ok=True)

    # 下载c.yaml
    c_path = os.path.join("node_files", "c.yaml")
    print(f"🔽 正在下载c.yaml到 {c_path}")
    with requests.get(c_yaml_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(c_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # 下载v.txt
    v_path = os.path.join("node_files", "v.txt")
    print(f"🔽 正在下载v.txt到 {v_path}")
    with requests.get(v_txt_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(v_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("\n🎉 全部完成！文件已保存在 node_files 文件夹")
    print(f"📁 c.yaml: {os.path.abspath(c_path)}")
    print(f"📁 v.txt: {os.path.abspath(v_path)}")

except Exception as e:
    print(f"❌ 出错了: {str(e)}")
