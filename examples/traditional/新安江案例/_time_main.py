"""包装 main.py, 输出总耗时。"""
import time, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 强制 UTF-8 输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 重定向到文件
log_file = open("_main_run.log", "w", encoding="utf-8")

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, s):
        for w in self.writers:
            w.write(s)
    def flush(self):
        for w in self.writers:
            try: w.flush()
            except: pass

sys.stdout = TeeWriter(sys.stdout, log_file)
sys.stderr = TeeWriter(sys.stderr, log_file)

t0 = time.perf_counter()
try:
    import main
    result = main.main()
except Exception:
    traceback.print_exc()
    raise
elapsed = time.perf_counter() - t0
print(f"\n[耗时] main() 总耗时: {elapsed:.1f} s ({elapsed/60:.2f} min)", flush=True)
log_file.close()
