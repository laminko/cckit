#!/usr/bin/env python3
"""Subprocess that ignores SIGTERM so transport.stop() must fall through to kill."""

import signal
import sys
import time


def main() -> None:
    # Ignore SIGTERM so terminate() has no effect
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
