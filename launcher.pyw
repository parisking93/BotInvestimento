import os, sys, json, subprocess, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.json"
SERVICE = "BotInvestimento"

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "defaults": {
            "pair": "BTC/EUR",
            "quote": "EUR",
            "ranges": "NOW,1H,24H,30D,90D,1Y",
            "max_pairs": 200,
            "public_qps": 1.6,
            "risk_level": 6,
            "budget_eur": 0.0,
            "reserve_eur": 0.0
        },
        "script": {
            "python": ".venv/Scripts/python.exe",
            "entry": "main.py"   # 👈 adesso punta a main.py
        }
    }

def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BotInvestimento - Launcher")
        self.geometry("720x500")
        self.proc = None

        self.cfg = load_config()
        d = self.cfg["defaults"]

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x", expand=False)

        row = 0
        def add(label, var):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
            e = ttk.Entry(frm, textvariable=var, width=42)
            e.grid(row=row, column=1, sticky="w", padx=6)
            row += 1
            return e

        self.pair_var     = tk.StringVar(value=d.get("pair","BTC/EUR"))
        self.quote_var    = tk.StringVar(value=d.get("quote","EUR"))
        self.ranges_var   = tk.StringVar(value=d.get("ranges","NOW,1H,24H,30D,90D,1Y"))
        self.maxpairs_var = tk.StringVar(value=str(d.get("max_pairs",200)))
        self.qps_var      = tk.StringVar(value=str(d.get("public_qps",1.6)))
        self.risk_var     = tk.StringVar(value=str(d.get("risk_level",6)))
        self.budget_var   = tk.StringVar(value=str(d.get("budget_eur",0.0)))
        self.reserve_var  = tk.StringVar(value=str(d.get("reserve_eur",0.0)))

        add("Pair (es. BTC/EUR)", self.pair_var)
        add("Quote (es. EUR)", self.quote_var)
        add("Ranges (comma)", self.ranges_var)
        add("Max pairs", self.maxpairs_var)
        add("Public QPS", self.qps_var)
        add("Risk level (1-10)", self.risk_var)
        add("Budget EUR", self.budget_var)
        add("Reserve EUR", self.reserve_var)

        btnfrm = ttk.Frame(frm)
        btnfrm.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Button(btnfrm, text="💾 Salva default", command=self.save_defaults).pack(side="left", padx=6)
        ttk.Button(btnfrm, text="▶️ Run", command=self.run_bot).pack(side="left", padx=6)
        ttk.Button(btnfrm, text="⏹ Stop", command=self.stop_bot).pack(side="left", padx=6)

        # area log
        self.log = scrolledtext.ScrolledText(self, wrap="word", height=15, state="disabled", background="#111", foreground="#0f0")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def save_defaults(self):
        self.cfg["defaults"] = {
            "pair": self.pair_var.get().strip() or "BTC/EUR",
            "quote": self.quote_var.get().strip() or "EUR",
            "ranges": self.ranges_var.get().strip() or "NOW,1H,24H,30D,90D,1Y",
            "max_pairs": int(float(self.maxpairs_var.get() or 200)),
            "public_qps": float(self.qps_var.get() or 1.6),
            "risk_level": int(float(self.risk_var.get() or 6)),
            "budget_eur": float(self.budget_var.get() or 0.0),
            "reserve_eur": float(self.reserve_var.get() or 0.0)
        }
        save_config(self.cfg)
        messagebox.showinfo("OK", "Default salvati")

    def run_bot(self):
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Attenzione", "Il bot è già in esecuzione.")
            return

        py = (ROOT / self.cfg["script"]["python"]).resolve()
        entry = (ROOT / self.cfg["script"]["entry"]).resolve()
        cmd = [str(py if py.exists() else sys.executable), str(entry)]

        env = os.environ.copy()
        env["BUDGET"] = self.budget_var.get()

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            threading.Thread(target=self._reader, daemon=True).start()
            self.append_log(f"== Avviato {entry.name} ==\n")
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def _reader(self):
        assert self.proc
        for line in self.proc.stdout:
            self.append_log(line)

    def stop_bot(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.append_log("== Processo terminato ==\n")
        else:
            messagebox.showinfo("Info", "Nessun bot in esecuzione.")

if __name__ == "__main__":
    App().mainloop()
