"""
并行评估:concurrent.futures.ProcessPoolExecutor + tqdm 进度条。

为什么用进程池:水文模型一般有 numpy 密集计算,GIL 不是瓶颈,但
fork 出多个解释器能让多核真正并行。线程池对 CPU 密集型无效。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def _resolve_n_jobs(n_jobs: int) -> int:
    """-1 表示用全部核;1 表示串行。"""
    if n_jobs == 1:
        return 1
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 or positive, got {n_jobs}")
    if n_jobs == -1:
        import os
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def _dummy_imports_for_pickle() -> None:
    """子进程需要这些 import 才能 unpickle 任务。

    ProcessPoolExecutor 在 spawn (Windows 默认) 启动时,子进程是干净解释器;
    必须从全局能看到所有用到的符号。
    """


def make_progress_bar(total: int, desc: str = "calibrating", disable: bool = False):
    """构造一个进度条;无 tqdm 时返回 None。

    使用方式::

        bar = make_progress_bar(total=n_starts, desc="multi-start")
        for future in as_completed(futures):
            res = future.result()
            if bar: bar.update(1)
        if bar: bar.close()
    """
    if not _TQDM or disable or total <= 1:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def parallel_map(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    n_jobs: int = 1,
    desc: str = "calibrating",
    show_progress: bool = True,
) -> List[Any]:
    """并行执行 func(item) over items,按原顺序返回结果。

    Args:
        func: 任务函数。必须可在子进程 pickle(全局可见的函数,不要 lambda/闭包)。
        items: 任务列表。
        n_jobs: 1=串行, -1=全核, >1=指定核数。
        desc: 进度条说明。
        show_progress: 是否显示进度条(无 tqdm 时自动降级为串行)。

    Returns:
        与 items 等长的结果列表(顺序对齐)。
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    if n_jobs == 1 or len(items) <= 1:
        # 串行
        return [func(x) for x in items]

    # 并行:不能用 lambda(无法 pickle)
    _dummy_imports_for_pickle()

    results: List[Any] = [None] * len(items)
    bar = make_progress_bar(total=len(items), desc=desc,
                            disable=not show_progress)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        future_to_idx = {ex.submit(func, x): i for i, x in enumerate(items)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    return results
