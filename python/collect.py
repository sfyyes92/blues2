# -*- coding: utf-8 -*-
import os
import re
import requests
from datetime import datetime
from urllib.parse import urljoin

# ===================== 自动生成今日节点页链接 =====================
def get_today_node_url():
    day = datetime.now().day  # 获取今天是几号（1~31）
    url = f"https://us1.zhuk.dpdns.org/{day}-Youtube.html"
    return url

# ===================== 提取 c.yaml 和 v.txt =====================
def extract_final_links(html):
    # 匹配 cYaml 和 vTxt
    c_match = re.search(r'id="cYaml"[^>]*>\s*(https?://[^\s<]+)', html, re.I)
    v_match = re.search(r'id="vTxt"[^>]*>\s*(https?://[^\s<]+)', html, re.I)

    if not c_match or not v_match:
        raise Exception("无法提取 c.yaml 和 v.txt")

    return c_match.group(1), v_match.group(1)

# ===================== 下载文件 =====================
def download_file(url, save_path):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)

# ===================== 主程序 =====================
def main():
    out_dir = "../output"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 自动生成今天的节点页
    node_url = get_today_node_url()
    print(f"今日节点页：{node_url}")

    # 2. 访问页面
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })
    html = session.get(node_url, timeout=15).text

    # 3. 直接提取最终链接
    yaml_url, txt_url = extract_final_links(html)
    print(f"c.yaml: {yaml_url}")
    print(f"v.txt:  {txt_url}")

    # 4. 下载
    download_file(yaml_url, os.path.join(out_dir, "c.yaml"))
    download_file(txt_url, os.path.join(out_dir, "v.txt"))

    print("\n✅ 全部下载完成！")
    print(f"文件保存在: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
