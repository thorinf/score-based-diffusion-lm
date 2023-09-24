"""
Logger reimplemented from OpenAI baselines:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import os.path as osp
import json
from collections import defaultdict
import logging

import wandb

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class SeqWriter(object):
    def write_seq(self, seq):
        raise NotImplementedError


class KVWriter(object):
    def write_kvs(self, kvs):
        raise NotImplementedError


class HumanOutputFormat(SeqWriter, KVWriter):
    def __init__(self, filename, overwrite=True, num_columns=4):
        mode = "wt" if overwrite else "at"
        if isinstance(filename, str):
            self.file = open(filename, mode)
            self.own_file = True
        else:
            assert hasattr(filename, "read"), "expected file or str, got %s" % filename
            self.file = filename
            self.own_file = False
        self.num_columns = num_columns

    def write_seq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def write_kvs(self, kvs):
        key2str = {}
        for key, val in kvs.items():
            if isinstance(val, (int, float)):
                valstr = f"{val:<8.3g}"
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if not key2str:
            return

        formatted_str = "\n".join(self._format_table(key2str, self.num_columns))
        self.file.write(formatted_str + "\n")
        self.file.flush()

    @staticmethod
    def _truncate(s, max_len=30):
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    @staticmethod
    def _format_table(kvs, n_columns=2):
        key_width = max(map(len, kvs.keys()))
        val_width = max(map(len, kvs.values()))

        sorted_items = sorted(kvs.items(), key=lambda kv: kv[0].lower())
        n_items = len(sorted_items)

        n_rows = (n_items + n_columns - 1) // n_columns
        sorted_items += [("", "")] * (n_rows * n_columns - n_items)

        lines = ["|" for _ in range(n_rows)]
        for i, (key, val) in enumerate(sorted_items):
            row = i % n_rows
            sep = ":" if key or val else " "
            line = f" {key.ljust(key_width)} {sep} {val.ljust(val_width)} |"
            lines[row] += line

        dashes = "-" * len(lines[0])
        lines.insert(0, dashes)
        lines.append(dashes)

        return lines

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename, overwrite=True):
        mode = "wt" if overwrite else "at"
        self.file = open(filename, mode)

    def write_kvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename, overwrite=True):
        mode = 'w+t' if overwrite else 'a+t'
        self.file = open(filename, mode)
        self.keys = []
        self.sep = ','

        if not overwrite:
            self.file.seek(0)
            header = self.file.readline().strip()
            if header:
                self.keys = header.split(self.sep)

    def write_kvs(self, kvs):
        extra_keys = sorted(list(kvs.keys() - self.keys))

        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)

            self.file.write(self.sep.join(self.keys) + '\n')

            for line in lines[1:]:
                padding = ["" for _ in extra_keys]
                updated_line = line.strip() + self.sep + self.sep.join(padding) + '\n'
                self.file.write(updated_line)

        values = [str(kvs.get(key, "")) for key in self.keys]
        self.file.write(self.sep.join(values) + '\n')

        self.file.flush()

    def close(self):
        self.file.close()


class WandBOutputFormat(KVWriter):
    def __init__(self):
        pass

    def write_kvs(self, kvs):
        data = kvs.copy()
        step = data.pop("step", None)
        if wandb.run is not None:
            wandb.log(kvs, step=step)


class Logger(object):
    CURRENT = None

    def __init__(self, output_dir, output_formats):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(float)
        self.level = INFO
        self.output_dir = output_dir
        self.output_formats = output_formats

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._log(args)

    def _log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.write_seq(map(str, args))

    def log_kv(self, key, value):
        self.name2val[key] = value

    def log_kv_mean(self, key, value, count: int = 1):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + count) + value * count / (cnt + count)
        self.name2cnt[key] = cnt + count

    def dump_kvs(self, step=None):
        if step is not None:
            self.log_kv("step", step)
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.write_kvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()


class RedirectLoggingHandler(logging.Handler):
    def __init__(self):
        super(RedirectLoggingHandler, self).__init__()

    def emit(self, record: logging.LogRecord) -> None:
        msg, level = self.format(record), record.levelno
        log(f"python logging: {msg}", level=level)


def configure_pylogging_handler():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    redirect_handler = RedirectLoggingHandler()
    logger.addHandler(redirect_handler)
    logger.info("redirecting to custom logger")


def configure(output_dir, overwrite=False):
    output_formats = [
        HumanOutputFormat(sys.stdout),
        HumanOutputFormat(osp.join(output_dir, "log.txt"), overwrite=overwrite),
        JSONOutputFormat(osp.join(output_dir, "log.json"), overwrite=overwrite),
        CSVOutputFormat(osp.join(output_dir, "log.csv"), overwrite=overwrite),
        WandBOutputFormat()
    ]

    Logger.CURRENT = Logger(output_dir=output_dir, output_formats=output_formats)

    configure_pylogging_handler()


def log(*args, level=INFO):
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def log_kv(key, val):
    get_current().log_kv(key, val)


def log_kv_mean(key, val, count=1):
    get_current().log_kv_mean(key, val, count)


def log_kvs(d):
    for (k, v) in d.items():
        log_kv(k, v)


def dump_kvs(step=None):
    return get_current().dump_kvs(step)


def get_current():
    return Logger.CURRENT
