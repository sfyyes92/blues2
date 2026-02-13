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


def download_to_path(session: requests.Session, url: str, out_path: str) -> str:
    """Download url to out_path (force file name)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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

    m = re.search(r'(https?://[^\s"<>]+/jmy/\d+\.html|/jmy/\d+\.html)', s)
    if m:
        return urljoin(base_url, m.group(1).strip())

    raise ValueError("未找到“点击解锁资源”按钮链接。")


# ===================== 5) 解密相关（保留你现有实现，不做增强） =====================

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


def decrypt(ciphertext, password):
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


# ===================== 6) 新增：明文中“直链/网页”两种模式解析，并固定命名下载 =====================

def extract_urls(text: str) -> List[str]:
    return re.findall(r'https?://[^\s"<>]+', text or "")


def normalize_url(u: str) -> str:
    u = (u or "").strip().strip('"').strip("'")
    # 订阅中心页里可能是 httpsus1...（缺少 ://）:contentReference[oaicite:2]{index=2}
    if u.startswith("https") and not u.startswith("https://"):
        u = u.replace("https", "https://", 1)
    if u.startswith("http") and not (u.startswith("http://") or u.startswith("https://")):
        u = u.replace("http", "http://", 1)
    return u


def extract_cv_urls_from_subscribe_page(html: str) -> Tuple[str, str]:
    """
    解析订阅中心网页：优先读取 id=cYaml 和 id=vTxt 的文本内容。
    你提供的源码里 vTxt 就是这样存的 :contentReference[oaicite:3]{index=3}
    """
    s = html

    # 精准取元素文本
    m_c = re.search(r'id\s*=\s*["\']cYaml["\'][^>]*>\s*([^<\s]+)\s*<', s, flags=re.I)
    m_v = re.search(r'id\s*=\s*["\']vTxt["\'][^>]*>\s*([^<\s]+)\s*<', s, flags=re.I)

    c_url = normalize_url(m_c.group(1)) if m_c else ""
    v_url = normalize_url(m_v.group(1)) if m_v else ""

    # 兜底：扫出任意 yaml/yml 与 txt
    if not c_url:
        m = re.search(r'(https?://[^\s"<>]+?\.(?:ya?ml))(?:\?|$)', s, flags=re.I)
        if m:
            c_url = normalize_url(m.group(1))
    if not v_url:
        m = re.search(r'(https?://[^\s"<>]+?\.txt)(?:\?|$)', s, flags=re.I)
        if m:
            v_url = normalize_url(m.group(1))

    if not c_url or not v_url:
        raise ValueError(f"订阅中心页未解析到 c/v 链接：c={c_url!r}, v={v_url!r}")

    return c_url, v_url


def resolve_cv_urls(session: requests.Session, plaintext: str) -> Tuple[str, str]:
    """
    明文可能：
    A) 直接包含订阅下载直链（.yaml/.yml + .txt）
    B) 不包含直链，只有一个网页链接，访问网页后从 id=cYaml / id=vTxt 取出链接
    """
    urls = extract_urls(plaintext)

    yaml_candidates = [u for u in urls if re.search(r'\.(?:ya?ml)(?:\?|$)', u, re.I)]
    txt_candidates = [u for u in urls if re.search(r'\.txt(?:\?|$)', u, re.I)]
    if yaml_candidates and txt_candidates:
        return yaml_candidates[0], txt_candidates[0]

    # 网页模式：优先试 .html 链接
    page_candidates = [u for u in urls if re.search(r'\.html?(?:\?|$)', u, re.I)]
    candidates = page_candidates or urls

    last_err = None
    for page_url in candidates:
        try:
            page_html = fetch_html(session, page_url)
            return extract_cv_urls_from_subscribe_page(page_html)
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"明文里未找到可直接下载的订阅链接，也无法从网页解析到订阅链接。最后错误：{last_err}")


# ===================== 7) 主流程 =====================

def main():
    channel_url = "https://www.youtube.com/@blue-Youtube"
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    session = make_session()

    # A) 频道主页 -> 最新日期视频
    channel_html = fetch_html(session, channel_url)
    video_url, dt, title = find_latest_dated_video_url_from_channel_html(channel_html)
    video_url = video_url.replace("www.youtube.com", "m.youtube.com")
    print("最新日期视频：")
    print("日期:", dt.isoformat())
    print("标题:", title)
    print("链接:", video_url)

    # B) 视频页 -> 节点下载页
    watch_html = fetch_html(session, video_url)
    try:
        node_page_url = extract_node_download_url_from_watch_html(watch_html)
    except Exception:
        dump_debug_html(watch_html, "watch_html")
        print("[DEBUG] watch_html length:", len(watch_html) if watch_html else 0)
        raise

    print("\n节点下载链接:", node_page_url)

    # C) 节点下载页 -> 解锁资源页
    node_page_html = fetch_html(session, node_page_url, extra_headers={"Referer": video_url})
    unlock_url = extract_unlock_resource_url_from_source(node_page_html, base_url=node_page_url)
    print("解锁资源链接:", unlock_url)

    # D) 解锁资源页 -> encryptedText -> 解密得到明文
    unlock_html = fetch_html(session, unlock_url, extra_headers={"Referer": node_page_url})
    encrypted_b64 = extract_encrypted_text_from_html(unlock_html)

    password = get_password(encrypted_b64)
    plaintext = decrypt_cryptojs_openssl_salted(encrypted_b64, password)

    # E) 新逻辑：明文里可能直链，也可能是订阅中心网页
    c_url, v_url = resolve_cv_urls(session, plaintext)

    print("\n最终待下载：")
    print(" - c:", c_url)
    print(" - v:", v_url)

    # F) 固定命名保存
    c_path = os.path.join(out_dir, "c.yaml")
    v_path = os.path.join(out_dir, "v.txt")

    print("\n开始下载并固定命名...")
    download_to_path(session, c_url, c_path)
    print("已下载 ->", c_path)
    download_to_path(session, v_url, v_path)
    print("已下载 ->", v_path)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
