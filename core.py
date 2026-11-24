import os, sys, json, time, threading, queue, hashlib, itertools, tempfile, urllib.request, re
import numpy as np

FROZEN = getattr(sys, "frozen", False)
BASE = os.path.dirname(sys.executable if FROZEN else os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
CACHE = os.path.join(BASE, "cache")
CLONES = os.path.join(BASE, "clones")
for _d in (MODELS, CACHE, CLONES):
    os.makedirs(_d, exist_ok=True)
SETTINGS = os.path.join(BASE, "settings.json")
LOGF = os.path.join(BASE, "log.txt")
BASE_VOICES = [("am_adam", "Adam (male)"), ("af_jessica", "Jessica (female)"), ("af_sarah", "Sarah (female)")]
REF_TEXT = ("The quick brown fox jumps over the lazy dog. Good morning, how are you today? "
            "I would like to talk with you about something important, so please give me a moment.")
PREVIEW_TEXT = "Hello, this is how my voice sounds."


def log(msg):
    try:
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- settings
def default_buttons():
    rows = [("Hello", "Hello, how are you?", "Greetings"), ("Yes", "Yes.", "Answers"), ("No", "No, thank you.", "Answers"),
            ("Okay", "Okay, sure.", "Answers"), ("One moment", "One moment please.", "Answers"),
            ("Repeat", "Could you please repeat that?", "Answers"), ("Thanks", "Thank you very much.", "Closing"),
            ("Goodbye", "Thank you, have a nice day. Goodbye.", "Closing")]
    return [{"id": "b%d" % i, "label": l, "text": t, "group": g} for i, (l, t, g) in enumerate(rows)]


def clean_buttons(bs):
    out, seen = [], set()
    for i, b in enumerate(bs or []):
        if not isinstance(b, dict) or not str(b.get("text", "")).strip():
            continue
        bid = str(b.get("id") or "n%d_%d" % (int(time.time()), i))
        if bid in seen:
            bid += "x%d" % i
        seen.add(bid)
        text = str(b["text"]).strip()[:400]
        out.append({"id": bid, "label": (str(b.get("label") or text)[:24]).strip(), "text": text,
                    "group": (str(b.get("group") or "General")[:20]).strip() or "General"})
    return out[:60]


def load_cfg():
    d = {"name": "", "company": "", "voice": "am_adam", "speed": 1.0, "cable": None, "monitor": None,
         "buttons": default_buttons(), "clones": []}
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            raw = json.load(f)
        for k in d:
            if k in raw:
                d[k] = raw[k]
        if not raw.get("buttons"):
            d["buttons"] = default_buttons()
    except FileNotFoundError:
        pass
    except Exception:
        try:
            os.replace(SETTINGS, SETTINGS + ".broken")
        except Exception:
            pass
        log("settings.json was damaged; started with defaults (old copy: settings.json.broken)")
    d["buttons"] = clean_buttons(d["buttons"])
    for k in ("cable", "monitor"):
        if not isinstance(d[k], str):
            d[k] = None
    try:
        d["speed"] = min(1.3, max(0.8, float(d["speed"])))
    except Exception:
        d["speed"] = 1.0
    return d


def save_cfg(cfg):
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    os.replace(tmp, SETTINGS)  # safe: file is never half-written


# ---------------------------------------------------------------- audio devices / playback
def list_outs():
    import sounddevice as sd
    pref = None
    for i, a in enumerate(sd.query_hostapis()):
        if "WASAPI" in a["name"]:
            pref = i
    if pref is None:
        pref = sd.default.hostapi
    return [{"index": i, "name": d["name"]} for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and d["hostapi"] == pref]


class Out:
    """One always-open output stream, so a button starts with no start-up delay."""

    def __init__(self, index):
        import sounddevice as sd
        info = sd.query_devices(index)
        self.sr = int(info["default_samplerate"])
        self.ch = max(1, min(2, info["max_output_channels"]))
        self.buf, self.pos, self.lock = None, 0, threading.Lock()
        self.stream = sd.OutputStream(device=index, samplerate=self.sr, channels=self.ch, dtype="float32",
                                      latency="low", callback=self.cb)
        self.stream.start()

    def cb(self, out, frames, t, status):
        with self.lock:
            b = self.buf
            if b is None:
                out.fill(0)
                return
            n = max(0, min(frames, len(b) - self.pos))
            if n:
                out[:n] = b[self.pos:self.pos + n, None]
            out[n:] = 0
            self.pos += n
            if self.pos >= len(b):
                self.buf = None

    def play(self, audio, sr):
        a = audio if sr == self.sr else np.interp(
            np.linspace(0, len(audio) - 1, int(len(audio) * self.sr / sr)), np.arange(len(audio)), audio).astype(np.float32)
        with self.lock:
            self.buf, self.pos = a, 0

    def stop(self):
        with self.lock:
            self.buf = None

    def progress(self):
        with self.lock:
            return (self.pos / len(self.buf)) if self.buf is not None and len(self.buf) else None

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


class Player:
    def __init__(self):
        self.outs, self.pid = [], None

    def reopen(self, indices):
        for o in self.outs:
            o.close()
        self.outs = []
        for i in dict.fromkeys(indices):
            self.outs.append(Out(i))

    def play(self, audio, sr, pid):
        self.pid = pid
        for o in self.outs:
            o.play(audio, sr)

    def stop(self):
        for o in self.outs:
            o.stop()

    def current(self):
        for o in self.outs:
            p = o.progress()
            if p is not None:
                return self.pid, p
        return None, 0


# ---------------------------------------------------------------- voice clone (OpenVoice v2 tone converter, free / MIT)
class Cloner:
    def __init__(self, eng):
        self.eng, self.conv, self.sr, self.torch, self.se = eng, None, 22050, None, {}

    def load(self):
        if self.conv:
            return
        import torch, huggingface_hub
        self.torch = torch
        d = os.path.join(MODELS, "openvoice")
        ck = os.path.join(d, "converter", "checkpoint.pth")
        if not os.path.exists(ck):
            self.eng.clone_msg = "Downloading clone model (only once)…"
            huggingface_hub.snapshot_download("myshell-ai/OpenVoiceV2", local_dir=d, allow_patterns=["converter/*"])
        from openvoice.api import ToneColorConverter
        cp = os.path.join(d, "converter", "config.json")
        try:
            c = ToneColorConverter(cp, device="cpu", enable_watermark=False)
        except TypeError:
            c = ToneColorConverter(cp, device="cpu")
        c.load_ckpt(ck)
        self.conv, self.sr = c, c.hps.data.sampling_rate

    def _se_from_wav(self, path):
        return self.conv.extract_se([path])

    def make(self, name):
        self.load()
        se = self._se_from_wav(os.path.join(CLONES, name + ".wav"))
        self.torch.save(se, os.path.join(CLONES, name + ".pt"))
        self.se.pop(name, None)

    def get_se(self, name):
        if name not in self.se:
            self.se[name] = self.torch.load(os.path.join(CLONES, name + ".pt"), map_location="cpu")
        return self.se[name]

    def base_se(self, voice):
        key = "_base_" + voice
        p = os.path.join(CLONES, key + ".pt")
        if key not in self.se:
            if not os.path.exists(p):
                a, sr = self.eng.kokoro(REF_TEXT, voice, 1.0)
                tmp = write_wav(a, sr)
                try:
                    self.torch.save(self._se_from_wav(tmp), p)
                finally:
                    os.remove(tmp)
            self.se[key] = self.torch.load(p, map_location="cpu")
        return self.se[key]

    def convert(self, audio, sr, base, name):
        self.load()
        src = self.base_se(base)
        tgt = self.get_se(name)
        tmp = write_wav(audio, sr)
        try:
            out = self.conv.convert(audio_src_path=tmp, src_se=src, tgt_se=tgt, tau=0.3)
        finally:
            os.remove(tmp)
        return np.asarray(out, dtype=np.float32), self.sr


def write_wav(a, sr):
    import soundfile as sf
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(p, a, sr)
    return p


# ---------------------------------------------------------------- engine: one worker thread, disk cache
class Item:
    def __init__(self):
        self.ev, self.audio, self.sr, self.err, self.persist = threading.Event(), None, 24000, None, False


class Engine:
    def __init__(self, app):
        self.app, self.k, self.err, self.status = app, None, "", "Loading voices…"
        self.q, self.seq = queue.PriorityQueue(), itertools.count()
        self.mem, self.pend, self.lock = {}, {}, threading.RLock()
        self.cloner = Cloner(self)
        self.clone_msg, self.clone_busy = "", False
        threading.Thread(target=self.run, daemon=True).start()

    # --- models
    def _find(self, names):
        for d in (MODELS, BASE):
            for n in names:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    def load_kokoro(self):
        from kokoro_onnx import Kokoro
        m = self._find(["kokoro.onnx", "kokoro-v1.0.onnx"])
        v = self._find(["voices.bin", "voices-v1.0.bin"])
        url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        if not m:
            self.status = "Downloading voice model (once)…"
            m = os.path.join(MODELS, "kokoro.onnx"); urllib.request.urlretrieve(url + "kokoro-v1.0.onnx", m)
        if not v:
            self.status = "Downloading voices (once)…"
            v = os.path.join(MODELS, "voices.bin"); urllib.request.urlretrieve(url + "voices-v1.0.bin", v)
        self.k = Kokoro(m, v)

    def run(self):
        try:
            self.load_kokoro()
            self.status = ""
        except Exception as e:
            self.err = "Voice engine failed: %s" % e
            log(self.err)
        while True:
            _, _, fn = self.q.get()
            try:
                fn()
            except Exception as e:
                log("job error: %r" % e)

    def submit(self, prio, fn):
        self.q.put((prio, next(self.seq), fn))

    # --- synthesis
    def kokoro(self, text, voice, speed):
        if not self.k:
            raise RuntimeError(self.err or "Voice engine not loaded yet")
        a, sr = self.k.create(text, voice=voice, speed=speed, lang="en-us")
        return np.asarray(a, dtype=np.float32), sr

    def synth(self, voice, speed, text):
        if voice.startswith("clone:"):
            name = voice[6:]
            base = self.app.clone_base(name)
            a, sr = self.kokoro(text, base, speed)
            return self.cloner.convert(a, sr, base, name)
        return self.kokoro(text, voice, speed)

    def key(self, voice, speed, text):
        vk = voice
        if voice.startswith("clone:"):
            try:
                vk += str(os.path.getmtime(os.path.join(CLONES, voice[6:] + ".pt")))
            except OSError:
                pass
        return hashlib.sha1(("%s|%.2f|%s" % (vk, speed, text)).encode()).hexdigest()[:20]

    def path(self, key):
        return os.path.join(CACHE, key + ".npy")

    def _save(self, key, it):
        p = self.path(key)
        if os.path.exists(p) or it.audio is None:
            return
        try:
            arr = np.concatenate([[it.sr], it.audio]).astype(np.float32)  # first value = sample rate
            with open(p + ".tmp", "wb") as f:
                np.save(f, arr)
            os.replace(p + ".tmp", p)
        except Exception as e:
            log("cache save failed: %r" % e)

    def ready(self, voice, speed, text):
        k = self.key(voice, speed, text)
        it = self.mem.get(k)
        return bool(it and it.ev.is_set() and not it.err) or os.path.exists(self.path(k))

    def ensure(self, voice, speed, text, persist=True, prio=5):
        k = self.key(voice, speed, text)
        with self.lock:
            it = self.mem.get(k)
            if it is None and os.path.exists(self.path(k)):
                try:
                    arr = np.load(self.path(k))
                    it = Item(); it.sr, it.audio = int(arr[0]), arr[1:].copy(); it.ev.set(); it.persist = True
                    self.mem[k] = it
                except Exception:
                    it = None
            if it is None:
                it = self.pend.get(k)
            if it is None:
                it = Item(); self.pend[k] = it
            it.persist = it.persist or persist
            if it.ev.is_set():
                if it.persist:
                    self._save(k, it)
            else:
                self.submit(prio, lambda: self._do(k, it, voice, speed, text))
        return it

    def _do(self, k, it, voice, speed, text):
        if it.ev.is_set():
            return
        t = time.time()
        try:
            it.audio, it.sr = self.synth(voice, speed, text)
        except Exception as e:
            it.err = str(e); log("synth error: %r" % e)
        with self.lock:
            self.pend.pop(k, None)
            if not it.err:
                self.mem[k] = it
        it.ev.set()
        if not it.err:
            if it.persist:
                self._save(k, it)
            log("   voice piece: %.2fs to make, %.2fs of audio | %s" % (time.time() - t, len(it.audio) / it.sr, text))

    # --- cloning jobs
    def add_clone(self, name, base):
        def job():
            try:
                self.clone_msg = "Learning your voice…"
                self.cloner.make(name)
                with self.app.lock:
                    c = [x for x in self.app.cfg["clones"] if x["name"] != name]
                    c.append({"name": name, "base": base})
                    self.app.cfg["clones"] = c
                    self.app.cfg["voice"] = "clone:" + name
                    save_cfg(self.app.cfg)
                self.clone_msg = "Voice “%s” is ready and selected." % name
                self.app.prep()
            except Exception as e:
                self.clone_msg = "Clone failed: %s" % e
                log("clone error: %r" % e)
            self.clone_busy = False
        self.clone_busy = True
        self.clone_msg = "Starting…"
        self.submit(1, job)


# ---------------------------------------------------------------- app state
class App:
    def __init__(self):
        self.lock = threading.RLock()
        self.cfg = load_cfg()
        self.eng, self.pl = Engine(self), Player()
        self.call, self.ticket, self.heard, self.err, self.loading = False, 0, [], "", None
        self.outs = []
        try:
            self.outs = list_outs()
        except Exception as e:
            self.err = "Audio problem: %s" % e
        c = self.cable_name()
        if c and not self.cfg["cable"]:
            self.cfg["cable"] = c
        self.reopen()
        self.prep()

    def cable_name(self):
        for o in self.outs:
            if "cable input" in o["name"].lower():
                return o["name"]
        return None

    def reopen(self):
        idx = [o["index"] for o in self.outs if o["name"] in (self.cfg["cable"], self.cfg["monitor"])]
        try:
            self.pl.reopen(idx)
            self.err = "" if idx else ("" if not self.outs else "Choose where the voice should go (Send voice to).")
        except Exception as e:
            self.err = "Could not open audio device: %s" % e

    def clone_base(self, name):
        for c in self.cfg["clones"]:
            if c["name"] == name:
                return c["base"]
        return "am_adam"

    def voices(self):
        v = [{"id": i, "name": n, "clone": False} for i, n in BASE_VOICES]
        v += [{"id": "clone:" + c["name"], "name": c["name"], "clone": True} for c in self.cfg["clones"]]
        return v

    def cur_voice(self):
        v = self.cfg["voice"]
        return v if any(x["id"] == v for x in self.voices()) else "am_adam"

    def intro(self):
        n, c = self.cfg["name"].strip(), self.cfg["company"].strip()
        who = "Hi, this is %s%s." % (n, " from " + c if c else "") if n else "Hi."
        return who + " I am using an AI generated voice."

    def prep(self, prio=3):
        v, sp = self.cur_voice(), self.cfg["speed"]
        self.eng.ensure(v, sp, self.intro(), True, prio)
        for b in self.cfg["buttons"]:
            self.eng.ensure(v, sp, b["text"], True, prio)

    def set_config(self, p):
        with self.lock:
            old = dict(self.cfg)
            for k in ("name", "company"):
                if k in p:
                    self.cfg[k] = str(p[k]).strip()[:60]
            if "voice" in p:
                self.cfg["voice"] = str(p["voice"])
            if "speed" in p:
                self.cfg["speed"] = min(1.3, max(0.8, float(p["speed"])))
            for k in ("cable", "monitor"):
                if k in p:
                    self.cfg[k] = p[k] if isinstance(p[k], str) and p[k] else None
            if "buttons" in p:
                self.cfg["buttons"] = clean_buttons(p["buttons"])
            save_cfg(self.cfg)
        if (old["cable"], old["monitor"]) != (self.cfg["cable"], self.cfg["monitor"]):
            self.reopen()
        old_ids = {b["id"]: b["text"] for b in old["buttons"]}
        v, sp = self.cur_voice(), self.cfg["speed"]
        for b in self.cfg["buttons"]:
            if old_ids.get(b["id"]) != b["text"]:
                self.eng.ensure(v, sp, b["text"], True, 1)  # new button: jump the queue
        self.prep()

    def speak(self, text, pid, persist=True, voice=None):
        with self.lock:
            self.ticket += 1
            t = self.ticket
        it = self.eng.ensure(voice or self.cur_voice(), self.cfg["speed"], text, persist, 0)
        if it.ev.is_set():
            self._play(it, text, pid)
            return
        self.loading = pid

        def wait():
            it.ev.wait()
            if self.loading == pid:
                self.loading = None
            if t == self.ticket:
                self._play(it, text, pid)
        threading.Thread(target=wait, daemon=True).start()

    def _play(self, it, text, pid):
        if it.err:
            self.err = it.err
            return
        self.err = "" if self.pl.outs else self.err
        self.pl.play(it.audio, it.sr, pid)
        self.heard = (self.heard + [text])[-4:]

    def press(self, bid):
        for b in self.cfg["buttons"]:
            if b["id"] == bid:
                return self.speak(b["text"], bid)

    def stop(self):
        self.ticket += 1
        self.loading = None
        self.pl.stop()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def set_call(self, on):
        self.call = bool(on)
        if on:
            self.speak(self.intro(), "intro")
        else:
            self.stop()

    def preview(self, voice):
        it = self.eng.ensure(voice, self.cfg["speed"], PREVIEW_TEXT, False, 0)

        def go():
            it.ev.wait()
            if it.err:
                self.err = it.err
                return
            try:
                import sounddevice as sd
                dev = next((o["index"] for o in self.outs if o["name"] == self.cfg["monitor"]), None)
                sd.play(it.audio, it.sr, device=dev)
            except Exception as e:
                self.err = "Preview failed: %s" % e
        threading.Thread(target=go, daemon=True).start()

    def clone_add(self, name, base, data):
        name = re.sub(r"[^\w \-]", "", name).strip()[:24]
        if not name:
            return {"error": "Give your voice a name."}
        if self.eng.clone_busy:
            return {"error": "Another clone is still being made."}
        if len(data) < 40000:
            return {"error": "Sample is too short. Record at least 10 seconds."}
        with open(os.path.join(CLONES, name + ".wav"), "wb") as f:
            f.write(data)
        self.eng.add_clone(name, base if base in dict(BASE_VOICES) else "am_adam")
        return {"ok": True}

    def clone_del(self, name):
        with self.lock:
            self.cfg["clones"] = [c for c in self.cfg["clones"] if c["name"] != name]
            if self.cfg["voice"] == "clone:" + name:
                self.cfg["voice"] = "am_adam"
            save_cfg(self.cfg)
        for ext in (".wav", ".pt"):
            try:
                os.remove(os.path.join(CLONES, name + ext))
            except OSError:
                pass
        self.prep()

    def state(self):
        c, v, sp = self.cfg, self.cur_voice(), self.cfg["speed"]
        ready = [b["id"] for b in c["buttons"] if self.eng.ready(v, sp, b["text"])]
        pid, prog = self.pl.current()
        e = self.eng
        return {"engine_ready": e.k is not None, "engine_status": e.status, "err": self.err or e.err,
                "cable_found": bool(self.cable_name()), "call": self.call, "loading": self.loading,
                "playing": {"id": pid, "progress": prog} if pid is not None else None,
                "voices": self.voices(), "outs": [o["name"] for o in self.outs],
                "cfg": {**c, "voice": v}, "ready": ready, "heard": self.heard, "intro": self.intro(),
                "clone": {"busy": e.clone_busy, "msg": e.clone_msg}}
