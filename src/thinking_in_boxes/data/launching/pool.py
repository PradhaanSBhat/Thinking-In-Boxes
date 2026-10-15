# Continuous worker pool: a free slot is refilled the moment one opens.

import os
import signal
import subprocess
import sys
import time
from datetime import datetime

_all_active_procs = []  # global reference for signal handler


def launch_one(task, log_dir):
    """Launch a single subprocess. Returns (Popen, task, log_file) or None."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, task["log_name"])
    log_f = open(log_path, "w")
    log_f.write(f"# {datetime.now().isoformat()}\n# {' '.join(task['cmd'])}\n\n")
    log_f.flush()

    env = os.environ.copy()
    env.update(task.get("env", {}))

    tmp = f"/tmp/blender_{task['log_name'].replace('.log', '')}"
    os.makedirs(tmp, exist_ok=True)
    env["TMPDIR"] = tmp
    env["OMP_NUM_THREADS"] = "2"

    try:
        p = subprocess.Popen(
            task["cmd"],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        print(f"  {task['label']}: PID {p.pid}")
        _all_active_procs.append((p, task, log_f))
        return (p, task, log_f)
    except Exception as e:
        print(f"  {task['label']}: FAILED: {e}")
        log_f.close()
        return None


def run_pool(all_tasks, max_concurrent, log_dir, stagger, check_fn=None, refresh=30):
    """Continuous pool – fills slots as workers finish, no wave boundaries.

    Instead of launching N workers, waiting for ALL to finish, then launching
    the next N, this keeps exactly max_concurrent workers running at all times.
    When any single worker finishes, the next task starts immediately.
    """
    t0 = time.time()
    task_queue = list(all_tasks)  # remaining tasks to launch
    active = []  # list of (Popen, task, log_file)
    all_results = []
    last_print = time.time()

    # Initial fill
    while task_queue and len(active) < max_concurrent:
        task = task_queue.pop(0)
        result = launch_one(task, log_dir)
        if result:
            active.append(result)
        if stagger > 0 and task_queue and len(active) < max_concurrent:
            time.sleep(stagger)

    # Main loop: poll for completions, fill slots
    while active:
        time.sleep(1)  # tight poll – check every second

        # Check for completed workers
        still_active = []
        for p, task, lf in active:
            rc = p.poll()
            if rc is not None:
                lf.close()
                all_results.append((task, rc))
                if rc != 0:
                    print(f"  {task['label']}: FAILED (exit {rc})")

                # immediately fill the slot with next task
                if task_queue:
                    next_task = task_queue.pop(0)
                    new_proc = launch_one(next_task, log_dir)
                    if new_proc:
                        still_active.append(new_proc)
            else:
                still_active.append((p, task, lf))

        active = still_active

        # Periodic status
        now = time.time()
        if now - last_print >= refresh:
            last_print = now
            elapsed = now - t0
            extra = check_fn() if check_fn else ""
            done = len(all_results)
            remaining = len(task_queue)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"active {len(active)}/{max_concurrent} | "
                f"done {done}/{len(all_tasks)} | "
                f"queued {remaining} | "
                f"{elapsed / 60:.1f}min{extra}"
            )

    return all_results


def report(results, t_start):
    total = time.time() - t_start
    failed = [(t, r) for t, r in results if r != 0]
    print(f"\n{'=' * 60}")
    print(f"  Time: {total:.0f}s ({total / 60:.1f}min)")
    print(f"  Workers: {len(results) - len(failed)} ok, {len(failed)} failed")
    if failed:
        for t, r in failed:
            print(f"    {t['label']}: exit {r}")
    print(f"{'=' * 60}")
    if failed:
        sys.exit(1)


def _setup_signal_handler():
    def _handler(_sig, _frame):
        _kill_all(_all_active_procs)

    signal.signal(signal.SIGINT, _handler)


def _kill_all(procs):
    print("\n[STOP] Killing workers ...")
    for p, _, _ in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
    time.sleep(3)
    for p, _, lf in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            lf.close()
        except Exception:
            pass
    sys.exit(130)
