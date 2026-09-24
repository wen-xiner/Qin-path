"""无头浏览器截图工具（走 Chrome DevTools Protocol）。

用途：本地起好前后端后，自动登录并对学生端 / 教师端截图，用于验收与文档配图。
依赖：本机 Edge（或 Chrome）+ websocket-client。

用法：
    python scripts/shot.py                     # 默认截学生端与教师端
    python scripts/shot.py --account teacher   # 只截教师端
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
PORT = 9333
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def find_browser() -> str:
    for p in EDGE_PATHS:
        if Path(p).exists():
            return p
    raise SystemExit("没找到 Edge 或 Chrome，请手动安装其一")


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=60)
        self._id = 0

    def send(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败：{msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str, await_promise: bool = False):
        return self.send(
            "Runtime.evaluate",
            expression=expr,
            awaitPromise=await_promise,
            returnByValue=True,
        )

    def shot(self, path: Path, full_page: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {"format": "png", "captureBeyondViewport": full_page}
        data = self.send("Page.captureScreenshot", **params)["data"]
        path.write_bytes(base64.b64decode(data))
        return path

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def wait_for(url: str, timeout: float = 30) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5173")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--account", default="both", choices=["both", "student", "teacher"])
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    args = ap.parse_args()

    if not wait_for(args.base):
        raise SystemExit(f"前端 {args.base} 没起来，请先 npm run dev")
    if not wait_for(f"{args.api}/api/health"):
        raise SystemExit(f"后端 {args.api} 没起来，请先起 uvicorn")

    browser = find_browser()
    profile = Path(__file__).resolve().parent / ".edge-profile"
    proc = subprocess.Popen(
        [
            browser,
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            # Chrome/Edge 111+ 默认拒绝来自任意 origin 的 CDP websocket 连接，
            # 不加这个会报 Handshake status 403 Forbidden
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            f"--window-size={args.width},{args.height}",
            "--hide-scrollbars",
            "--no-first-run",
            "--disable-gpu",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ws_url = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2) as r:
                    targets = json.load(r)
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        if ws_url is None:
            raise SystemExit("连不上无头浏览器调试端口")

        cdp = CDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")

        plans = []
        if args.account in ("both", "student"):
            plans.append(("student", "stu01", "学生端"))
        if args.account in ("both", "teacher"):
            plans.append(("teacher", "teacher", "教师端"))

        for role, username, label in plans:
            print(f"→ 登录 {username} 并截图（{label}）")
            cdp.send("Page.navigate", url=args.base)
            time.sleep(3.0)
            expr = f"""
            (async () => {{
              const r = await fetch('/api/auth/login', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{username: '{username}', password: '123456'}})
              }});
              const d = await r.json();
              localStorage.setItem('zhitu_token', d.access_token);
              localStorage.setItem('zhitu_user', JSON.stringify(d.user));
              return d.user.role;
            }})()
            """
            res = cdp.eval(expr, await_promise=True)
            role_name = res.get("result", {}).get("value", role)
            cdp.send("Page.reload", ignoreCache=True)
            # 等数据加载完成：学生端要跑 28 个知识点的 BKT 推断，留足时间
            time.sleep(12.0)
            out = OUT_DIR / f"{role}.png"
            cdp.shot(out)
            print(f"  角色={role_name}  已保存 {out}")

        cdp.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
