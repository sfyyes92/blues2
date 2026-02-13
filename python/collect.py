# -*- coding: utf-8 -*-
import os
import re
import json
import time
import random
import base64
import hashlib
import logging
import urllib.parse
from datetime import date
from typing import Any, Dict, Iterator, List, Optional, Tuple
from html import unescape
from urllib.parse import urljoin, urlparse, unquote

import requests
from Crypto.Cipher import AES
from Crypto.Hash import MD5
from Crypto.Util.Padding import unpad

import os
from datetime import datetime

def dump_debug_html(html: str, filename_prefix: str):
    """
    将 HTML 保存到 ../output/ 目录
    """
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{filename_prefix}_{ts}.html")

    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(html or "")

    print(f"[DEBUG] HTML saved to: {path}")
    return path



# ===================== 1) 通用：HTTP 请求 =====================

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    return s


def fetch_html(session: requests.Session, url: str, timeout: int = 25, extra_headers: Optional[dict] = None) -> str:
    """Fetch HTML with optional per-request headers (useful for Referer)."""
    time.sleep(random.uniform(0.2, 0.8))
    headers = session.headers.copy()
    if extra_headers:
        headers.update(extra_headers)
    r = session.get(url, timeout=timeout, allow_redirects=True, headers=headers)
    r.raise_for_status()
    return r.text


def download_file(session: requests.Session, url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path) or "download.bin"
    out_path = os.path.join(out_dir, filename)

    time.sleep(random.uniform(0.2, 0.6))
    r = session.get(url, timeout=40, allow_redirects=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


# ===================== 2) 从频道主页找“标题以日期打头”的最新视频 =====================

DATE_RE = re.compile(r"^\s*(\d{4})年(\d{1,2})月(\d{1,2})日")


def extract_yt_initial_data(html: str) -> Dict[str, Any]:
    patterns = [
        r"var\s+ytInitialData\s*=\s*(\{.*?\})\s*;\s*</script>",
        r'window\["ytInitialData"\]\s*=\s*(\{.*?\})\s*;\s*</script>',
        r"ytInitialData\s*=\s*(\{.*?\})\s*;\s*</script>",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
    raise ValueError("未在频道页 HTML 中找到 ytInitialData（可能拿到同意页/结构变化）。")


def iter_any_renderer(obj: Any, keys: Tuple[str, ...]) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, dict):
                yield v
        for v in obj.values():
            yield from iter_any_renderer(v, keys)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_any_renderer(it, keys)


def get_title_text(renderer: Dict[str, Any]) -> str:
    title = renderer.get("title", {})
    if isinstance(title, dict):
        if "simpleText" in title:
            return title["simpleText"]
        runs = title.get("runs")
        if isinstance(runs, list) and runs:
            return "".join(r.get("text", "") for r in runs if isinstance(r, dict))
    return ""


def parse_leading_date(title: str) -> Optional[date]:
    m = DATE_RE.match(title)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return date(y, mo, d)


def find_latest_dated_video_url_from_channel_html(channel_html: str) -> Tuple[str, date, str]:
    data = extract_yt_initial_data(channel_html)

    # 频道主页常见：gridVideoRenderer；也兼容 videoRenderer
    renderer_keys = ("gridVideoRenderer", "videoRenderer")

    candidates: List[Tuple[date, str, str]] = []
    for r in iter_any_renderer(data, renderer_keys):
        vid = r.get("videoId")
        if not vid:
            continue
        title = get_title_text(r)
        dt = parse_leading_date(title)
        if dt is None:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        candidates.append((dt, url, title))

    if not candidates:
        raise ValueError("频道页中没找到任何“标题以日期打头”的视频。")

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dt, latest_url, latest_title = candidates[0]
    return latest_url, latest_dt, latest_title


# ===================== 3) 从视频页简介提取“节点下载：”链接 =====================

def extract_yt_initial_player_response(html: str) -> Dict[str, Any]:
    """Extract ytInitialPlayerResponse JSON from a YouTube watch page."""
    patterns = [
        r"var\s+ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;\s*</script>",
        r"ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;\s*</script>",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
    raise ValueError("未在 watch HTML 中找到 ytInitialPlayerResponse（可能拿到同意页/结构变化）。")

def extract_node_download_url_from_watch_html(watch_html: str) -> str:
    """
    从 YouTube watch HTML 中提取“节点下载：”后的链接。
    兼容两种情况：
    1) structuredDescription 里直接出现完整直链
    2) 直链被截断成 ...，但在 youtube.com/redirect 的 q= 里仍有完整链接
    """
    # 1) 先做最关键的转义还原：\u0026 -> &，以及 \/ -> /
    s = watch_html
    s = s.replace("\\u0026", "&").replace("\\u003d", "=").replace("\\/", "/")
    s = unescape(s)

    # 2) 先尝试：简介里直接抓“节点下载：<URL>”
    url_charset = r"[A-Za-z0-9\-\._~:/\?#\[\]@!\$&'\(\)\*\+,;=%]+"
    m = re.search(rf"节点下载：\s*(https?://{url_charset})", s)
    if m:
        direct = m.group(1).strip()
        # GitHub 那份 HTML 常见截断：... 或者以 . 结尾但不是 .html
        if ("..." not in direct) and (direct.lower().endswith(".html")):
            return direct

    # 3) 兜底：从 youtube redirect 链接里解析 q=
    # 例子：...redirect?event=video_description&redir_token=...&q=https%3A%2F%2Fus1.zhuk.dpdns.org%2F12-Youtube.html&v=...
    candidates = []
    for mm in re.finditer(r"https?://www\.youtube\.com/redirect\?[^\"'\s<>]+", s):
        redir = mm.group(0)
        # 再确保一次转义（防止局部没被替换到）
        redir = redir.replace("\\u0026", "&").replace("\\u003d", "=").replace("\\/", "/")

        try:
            parsed = urlparse(redir)
            qs = urllib.parse.parse_qs(parsed.query)
            qv = qs.get("q", [None])[0]
            if not qv:
                continue
            real = urllib.parse.unquote(qv).strip()

            # 只收集看起来像目标的链接
            # 你例子是 https://us1.zhuk.dpdns.org/12-Youtube.html
            if real.startswith("http") and ("zhuk.dpdns.org" in real):
                candidates.append(real)
        except Exception:
            continue

    # 优先返回最像“节点下载页”的那个
    for u in candidates:
        if u.lower().endswith("-youtube.html") or u.lower().endswith("youtube.html"):
            return u

    # 再次兜底：任何包含 12-Youtube.html 的
    for u in candidates:
        if "Youtube.html" in u or "youtube.html" in u:
            return u

    raise ValueError("未找到完整节点下载链接（可能被 YouTube 截断或结构变动）")



# ===================== 4) 从 12-Youtube.html 页面提取“点击解锁资源”按钮链接 =====================

def extract_unlock_resource_url_from_source(html: str, base_url: str) -> str:
    s = unescape(html)

    m = re.search(
        r'<a[^>]*href="([^"]+)"[^>]*>\s*点击解锁资源\s*</a>',
        s,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return urljoin(base_url, m.group(1).strip())

    # 兜底：找 /jmy/xx.html
    m = re.search(r'(https?://[^\s"<>]+/jmy/\d+\.html|/jmy/\d+\.html)', s)
    if m:
        return urljoin(base_url, m.group(1).strip())

    raise ValueError("未找到“点击解锁资源”按钮链接。")


# ===================== 5) 解密 jmy/12.html 的 encryptedText（已知密码/占位） =====================

def extract_encrypted_text_from_html(html: str) -> str:
    m = re.search(r'const\s+encryptedText\s*=\s*"([^"]+)"\s*;', html, flags=re.DOTALL)
    if not m:
        raise ValueError("未找到 encryptedText")
    return m.group(1).strip()


def evp_bytes_to_key_md5(password: bytes, salt: bytes, key_len: int, iv_len: int):
    d = b""
    last = b""
    while len(d) < key_len + iv_len:
        last = hashlib.md5(last + password + salt).digest()
        d += last
    return d[:key_len], d[key_len:key_len + iv_len]


def decrypt_cryptojs_openssl_salted(ciphertext_b64: str, password: str) -> str:
    raw = base64.b64decode(ciphertext_b64)
    if raw[:8] != b"Salted__":
        raise ValueError("不是 OpenSSL Salted__ 格式（开头不是 'Salted__'）")
    salt = raw[8:16]
    ct = raw[16:]

    key, iv = evp_bytes_to_key_md5(password.encode("utf-8"), salt, 32, 16)  # AES-256-CBC
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = cipher.decrypt(ct)

    pad = pt[-1]
    if pad < 1 or pad > 16:
        raise ValueError("解密失败：padding 异常（密码可能不对）")
    pt = pt[:-pad]

    # 网页里一般会 decodeURIComponent
    return unquote(pt.decode("utf-8", errors="strict"))

def decrypt(ciphertext, password):
    """Decrypts the given ciphertext using the provided password."""
    try:
        encrypt_data = base64.b64decode(ciphertext)
        salt = encrypt_data[8:16]
        ciphertext = encrypt_data[16:]

        # Derive the key and IV
        derived = b""
        while len(derived) < 48:
            hasher = MD5.new()
            hasher.update(derived[-16:] + password.encode("utf-8") + salt)
            derived += hasher.digest()

        key, iv = derived[:32], derived[32:48]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), 16)
        return decrypted.decode("utf-8")
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")

def brute_force_password(encrypted_data):
    """Attempts to brute-force the password to decrypt the data."""
    print("start brute_force_password")
    for pwd in range(0, 10000):
        try:
            password = f"{pwd:04d}"
            decoded_data = decrypt(encrypted_data, str(password))
            decrypted_data = urllib.parse.unquote(decoded_data)
            logging.info(password)
            print("password: ",password)
            return password
        except ValueError:
            continue
    raise ValueError("Failed to brute-force the encryption password.")


def get_password(ciphertext_b64: str) -> str:
    return brute_force_password(ciphertext_b64)


def extract_urls(text: str) -> List[str]:
    return re.findall(r'https?://[^\s"<>]+', text)


def pick_c_yaml_and_v_txt(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        if u.endswith("c.yaml") or u.endswith("/c.yaml"):
            out.append(u)
        elif u.endswith("v.txt") or u.endswith("/v.txt"):
            out.append(u)
    # 去重保持顺序
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# ===================== 6) 主流程：从 YouTube 主页开始到下载文件 =====================

def main():
    channel_url = "https://www.youtube.com/@blue-Youtube"
    out_dir = "downloads"

    session = make_session()

    # A) 访问频道主页 -> 找最新日期打头的视频
    channel_html = fetch_html(session, channel_url)
    video_url, dt, title = find_latest_dated_video_url_from_channel_html(channel_html)
    video_url = video_url.replace("www.youtube.com", "m.youtube.com")
    print("最新日期视频：")
    print("日期:", dt.isoformat())
    print("标题:", title)
    print("链接:", video_url)

    # B) 访问视频页 -> 提取“节点下载：”链接（12-Youtube.html）
    watch_html = fetch_html(session, video_url)

    try:
        node_page_url = extract_node_download_url_from_watch_html(watch_html)
    except Exception as e:
        dump_debug_html(watch_html, "watch_html")
        print("[DEBUG] watch_html length:", len(watch_html) if watch_html else 0)
        raise

    print("\n节点下载链接:", node_page_url)
    print("node_page_url repr:", repr(node_page_url))

    # C) 访问 12-Youtube.html -> 提取“点击解锁资源”按钮链接（jmy/12.html）
    node_page_html = fetch_html(session, node_page_url, extra_headers={"Referer": video_url})
    unlock_url = extract_unlock_resource_url_from_source(node_page_html, base_url=node_page_url)
    print("解锁资源链接:", unlock_url)

    # D) 访问 jmy/12.html -> 提取 encryptedText -> 解密得到明文 HTML
    unlock_html = fetch_html(session, unlock_url, extra_headers={"Referer": node_page_url})
    encrypted_b64 = extract_encrypted_text_from_html(unlock_html)

    # 你本地已有实现的话，把 brute_force_password() 和 get_password() 按你逻辑调整回去即可
    password = get_password(encrypted_b64)
    plaintext = decrypt_cryptojs_openssl_salted(encrypted_b64, password)

    # E) 从明文中抓 URL -> 只下载 c.yaml 和 v.txt
    urls = extract_urls(plaintext)
    targets = pick_c_yaml_and_v_txt(urls)
    if not targets:
        raise ValueError("解密明文中未找到 c.yaml 或 v.txt 链接")

    print("\n待下载文件：")
    for u in targets:
        print(" -", u)

    print("\n开始下载...")
    for u in targets:
        path = download_file(session, u, out_dir=out_dir)
        print(f"已下载: {u}\n  -> {path}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
