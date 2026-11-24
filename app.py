import json, os, sys, threading, webbrowser, subprocess, zipfile, urllib.request, urllib.parse, tempfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import core

PORT = 8765
URL = "http://127.0.0.1:%d" % PORT
APP = None


def index_html():
    for d in (core.BASE, getattr(sys, "_MEIPASS", ""), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, "index.html")
        if d and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    return b"index.html not found"


def install_cable():
    try:
        tmp = tempfile.mkdtemp()
        z = os.path.join(tmp, "cable.zip")
        urllib.request.urlretrieve("https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip", z)
        zipfile.ZipFile(z).extractall(tmp)
        import ctypes
        ctypes.windll.shell32.ShellExecuteW(None, "runas", os.path.join(tmp, "VBCABLE_Setup_x64.exe"), None, tmp, 1)
    except Exception:
        webbrowser.open("https://vb-audio.com/Cable/")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def trusted(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in ("127.0.0.1:%d" % PORT, "localhost:%d" % PORT) and (origin in (None, URL, "http://localhost:%d" % PORT))

    def do_GET(self):
        if not self.trusted():
            return self.send({"error": "forbidden"}, 403)
        if self.path == "/api/state":
            return self.send(APP.state())
        self.send(index_html(), 200, "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not self.trusted():
            return self.send({"error": "forbidden"}, 403)
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/api/clone":
                return self.send(APP.clone_add(q.get("name", [""])[0], q.get("base", ["am_adam"])[0], raw))
            b = json.loads(raw or b"{}")
            p = u.path
            if p == "/api/press": APP.press(b.get("id"))
            elif p == "/api/say": APP.speak(str(b.get("text", ""))[:400], "say", persist=False)
            elif p == "/api/prepare":
                t = str(b.get("text", "")).strip()[:400]
                if t: APP.eng.ensure(APP.cur_voice(), APP.cfg["speed"], t, False, 2)
            elif p == "/api/stop": APP.stop()
            elif p == "/api/call": APP.set_call(b.get("on"))
            elif p == "/api/config": APP.set_config(b)
            elif p == "/api/preview": APP.preview(str(b.get("voice", "am_adam")))
            elif p == "/api/clone_del": APP.clone_del(str(b.get("name", "")))
            elif p == "/api/install_cable": threading.Thread(target=install_cable, daemon=True).start()
            else: return self.send({"error": "unknown"}, 404)
            self.send({"ok": True})
        except Exception as e:
            core.log("api error: %r" % e)
            self.send({"error": str(e)}, 500)


def main():
    global APP
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    except OSError:
        webbrowser.open(URL)  # already running
        return
    srv.daemon_threads = True
    APP = core.App()
    print("AccentLive Pro is running at", URL)
    print("Keep this window open. Close it to quit.")
    threading.Timer(1.0, lambda: webbrowser.open(URL)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
