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


def download_bytes(session: requests.Session, url: str) -> bytes:
    time.sleep(random.uniform(0.2, 0.6))
    r = session.get(url, timeout=40, allow_redirects=True)
    r.raise_for_status()
    return r.content


def download_file(session: requests.Session, url: str, out_dir: str, filename_override: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)

    if filename_override:
        filename = filename_override
    else:
        filename = os.path.basename(urlparse(url).path) or "download.bin"

    out_path = os.path.join(out_dir, filename)
    content = download_bytes(session, url)

    with open(out_path, "wb") as f:
        f.write(content)

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
    s = watch_html
    s = s.replace("\\u0026", "&").replace("\\u003d", "=").replace("\\/", "/")
    s = unescape(s)

    url_charset = r"[A-Za-z0-9\-\._~:/\?#\[\]@!\$&'\(\)\*\+,;=%]+"
    m = re.search(rf"节点下载：\s*(https?://{url_charset})", s)
    if m:
        direct = m.group(1).strip()
        if ("..." not in direct) and (direct.lower().endswith(".html")):
            return direct

    candidates = []
    for mm in re.finditer(r"https?://www\.youtube\.com/redirect\?[^\"'\s<>]+", s):
        redir = mm.group(0)
        redir = redir.replace("\\u0026", "&").replace("\\u003d", "=").replace("\\/", "/")

        try:
            parsed = urlparse(redir)
            qs = urllib.parse.parse_qs(parsed.query)
            qv = qs.get("q", [None])[0]
            if not qv:
                continue
            real = urllib.parse.unquote(qv).strip()
            if real.startswith("http") and ("zhuk.dpdns.org" in real):
                candidates.append(real)
        except Exception:
            continue

    for u in candidates:
        if u.lower().endswith("-youtube.html") or u.lower().endswith("youtube.html"):
            return u

    for u in candidates:
        if "Youtube.html" in u or "youtube.html" in u:
            return u

    raise ValueError("未找到完整节点下载链接（可能被 YouTube 截断或结构变动）")


# ===================== 4) 新增：从“节点下载页(12-Youtube.html)”直接提取最终订阅链接 =====================

def _normalize_possible_url(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()

    # 正常情况：已经是 http(s)://
    if s.startswith("http://") or s.startswith("https://"):
        return s

    # 某些“被压扁”的源码里可能出现： httpsus1.zhuk.dpdns.org/xxx
    # 或者： httpsus1.zhuk.dpdns.org202601cv12...
    if s.startswith("https") and "://" not in s:
        s = "https://" + s[len("https"):]
        return s
    if s.startswith("http") and "://" not in s:
        s = "http://" + s[len("http"):]
        return s

    return None


def extract_final_links_from_node_page_html(node_page_html: str) -> Optional[Tuple[str, str]]:
    """
    兼容你新上传的这种页面：源码里直接有
      <div id="cYaml">https://.../xxx.yaml</div>
      <div id="vTxt">https://.../yyy.txt</div>
    直接返回 (yaml_url, txt_url)，否则返回 None。

    这个结构在你的 source.txt 里就是这样放的。 :contentReference[oaicite:1]{index=1}
    """
    s = unescape(node_page_html or "")

    # 优先匹配 HTML 原样（含 https://）
    m1 = re.search(r'id="cYaml"[^>]*>\s*([^<\s]+)\s*<', s, flags=re.I)
    m2 = re.search(r'id="vTxt"[^>]*>\s*([^<\s]+)\s*<', s, flags=re.I)

    c = _normalize_possible_url(m1.group(1)) if m1 else None
    v = _normalize_possible_url(m2.group(1)) if m2 else None

    if c and v:
        return c, v

    # 再兜底一次：如果页面不是标准 HTML（比如被你保存时丢了 <>），就直接找关键片段附近的 https
    # （尽量保守：仍只抓 yaml/txt）
    m1b = re.search(r'cYaml.*?(https[^\s"<>\']+)', s, flags=re.I | re.S)
    m2b = re.search(r'vTxt.*?(https[^\s"<>\']+)', s, flags=re.I | re.S)

    c2 = _normalize_possible_url(m1b.group(1)) if m1b else None
    v2 = _normalize_possible_url(m2b.group(1)) if m2b else None
    if c2 and v2:
        return c2, v2

    return None


# ===================== 5) 从 12-Youtube.html 页面提取“点击解锁资源”按钮链接 =====================

def extract_unlock_resource_url_from_source(html: str, base_url: str) -> str:
    s = unescape(html)

    m = re.search(
        r'<a[^>]*href="([^"]+)"[^>]*>\s*点击解锁资源\s*</a>',
        s,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return urljoin(base_url, m.group(1).strip())

    m = re.search(r'(https?://[^\s"<>]+/jmy/\d+\.html|/jmy/\d+\.html)', s)
    if m:
        return urljoin(base_url, m.group(1).strip())

    raise ValueError("未找到“点击解锁资源”按钮链接。")


# ===================== 6) 解密 jmy/12.html 的 encryptedText =====================

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

    key, iv = evp_bytes_to_key_md5(password.encode("utf-8"), salt, 32, 16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = cipher.decrypt(ct)

    pad = pt[-1]
    if pad < 1 or pad > 16:
        raise ValueError("解密失败：padding 异常（密码可能不对）")
    pt = pt[:-pad]

    return unquote(pt.decode("utf-8", errors="strict"))


# ===== 你原来的 brute_force / decrypt，按你写的保留 =====

def decrypt(ciphertext, password):
    """Decrypts the given ciphertext using the provided password."""
    try:
        encrypt_data = base64.b64decode(ciphertext)
        salt = encrypt_data[8:16]
        ciphertext = encrypt_data[16:]

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
            print("password: ", password)
            return password
        except ValueError:
            continue
    raise ValueError("Failed to brute-force the encryption password.")


def get_password(ciphertext_b64: str) -> str:
    return brute_force_password(ciphertext_b64)


# ===================== 7) 提取链接 + 选择要下载的 yaml/txt（文件名可变） =====================

def extract_urls(text: str) -> List[str]:
    return re.findall(r'https?://[^\s"<>]+', text or "")


def pick_yaml_and_txt(urls: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    不再强依赖 c.yaml / v.txt 文件名。
    - yaml: 取第一个 .yaml 或 .yml
    - txt: 取第一个 .txt
    """
    yaml_url = None
    txt_url = None

    for u in urls:
        ul = u.lower()
        if yaml_url is None and (ul.endswith(".yaml") or ul.endswith(".yml")):
            yaml_url = u
        if txt_url is None and ul.endswith(".txt"):
            txt_url = u
        if yaml_url and txt_url:
            break

    return yaml_url, txt_url


# ===================== 8) 主流程：从 YouTube 主页开始到下载文件 =====================

def main():
    channel_url = "https://www.youtube.com/@blue-Youtube"
    out_dir = "../output/"

    session = make_session()

    # A) 访问频道主页 -> 找最新日期打头的视频
    channel_html = fetch_html(session, channel_url)
    video_url, dt, title = find_latest_dated_video_url_from_channel_html(channel_html)
    video_url = video_url.replace("www.youtube.com", "m.youtube.com")

    print("最新日期视频：")
    print("日期:", dt.isoformat())
    print("标题:", title)
    print("链接:", video_url)

    # B) 访问视频页 -> 提取“节点下载：”链接（xx-Youtube.html）
    watch_html = fetch_html(session, video_url)

    try:
        node_page_url = extract_node_download_url_from_watch_html(watch_html)
    except Exception:
        dump_debug_html(watch_html, "watch_html")
        print("[DEBUG] watch_html length:", len(watch_html) if watch_html else 0)
        raise

    print("\n节点下载链接:", node_page_url)
    print("node_page_url repr:", repr(node_page_url))

    # C) 访问 节点下载页（xx-Youtube.html）
    node_page_html = fetch_html(session, node_page_url, extra_headers={"Referer": video_url})

    # ===== 新增兼容：如果节点下载页里已经直接给出了最终订阅链接，就直接下载并结束 =====
    direct = extract_final_links_from_node_page_html(node_page_html)
    if direct:
        yaml_url, txt_url = direct
        print("\n[直达] 节点页已包含最终订阅链接：")
        print(" - YAML:", yaml_url)
        print(" - TXT :", txt_url)

        print("\n开始下载...")
        p1 = download_file(session, yaml_url, out_dir=out_dir, filename_override="c.yaml")
        print(f"已下载: {yaml_url}\n  -> {p1}")
        p2 = download_file(session, txt_url, out_dir=out_dir, filename_override="v.txt")
        print(f"已下载: {txt_url}\n  -> {p2}")

        print("\n全部完成。")
        return

    # ===== 原逻辑：节点页 -> 找“点击解锁资源” -> 解密 -> 提取下载链接 =====
    unlock_url = extract_unlock_resource_url_from_source(node_page_html, base_url=node_page_url)
    print("解锁资源链接:", unlock_url)

    unlock_html = fetch_html(session, unlock_url, extra_headers={"Referer": node_page_url})
    encrypted_b64 = extract_encrypted_text_from_html(unlock_html)

    password = get_password(encrypted_b64)
    plaintext = decrypt_cryptojs_openssl_salted(encrypted_b64, password)

    urls = extract_urls(plaintext)
    yaml_url, txt_url = pick_yaml_and_txt(urls)

    # 兼容：明文里可能没有直链，而是“网页访问链接”，再访问一次拿到直链（你之前的 down_link_sourcecode 就是这种）
    if not (yaml_url and txt_url):
        # 找一个看起来像“页面”的链接再访问
        page_url = None
        for u in urls:
            ul = u.lower()
            if ul.endswith(".html") or ul.endswith("/") or "dpdns.org" in ul:
                page_url = u
                break
        if page_url:
            try:
                page_html = fetch_html(session, page_url, extra_headers={"Referer": unlock_url})
                direct2 = extract_final_links_from_node_page_html(page_html)
                if direct2:
                    yaml_url, txt_url = direct2
            except Exception:
                pass

    if not (yaml_url and txt_url):
        raise ValueError("未能在解密明文/二次页面中找到 yaml 与 txt 下载链接")

    print("\n待下载文件：")
    print(" -", yaml_url)
    print(" -", txt_url)

    print("\n开始下载...")
    p1 = download_file(session, yaml_url, out_dir=out_dir, filename_override="c.yaml")
    print(f"已下载: {yaml_url}\n  -> {p1}")
    p2 = download_file(session, txt_url, out_dir=out_dir, filename_override="v.txt")
    print(f"已下载: {txt_url}\n  -> {p2}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
