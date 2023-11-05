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
            valstr = f"{val:<8.3g}" if isinstance(val, float) else str(val)
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
        self.file = open(filename, "a+")
        self.keys = []
        self.sep = ","

        if overwrite:
            self.file.seek(0)
            self.file.truncate()
        else:
            self.file.seek(0)
            header = self.file.readline().strip()
            if header:
                self.keys = header.split(self.sep)

    def write_kvs(self, kvs):
        new_keys = sorted(list(kvs.keys() - self.keys))

        if new_keys:
            self.keys.extend(new_keys)
            self.file.seek(0)
            lines = self.file.readlines()

            header = self.sep.join(self.keys) + '\n'
            if lines:
                lines[0] = header
            else:
                lines = [header]

            num_keys = len(self.keys)
            for i in range(1, len(lines)):
                values = lines[i].strip().split(self.sep)
                lines[i] = self.sep.join(values + [""] * (num_keys - len(values))) + '\n'

            # a crash between truncate and writing will lose csv data
            self.file.seek(0)
            self.file.truncate()
            self.file.writelines(lines)

        values = [str(kvs.get(key, "")) for key in self.keys]
        self.file.write(self.sep.join(values) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, log_dir):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            error("Unable to import tensorboard. Please ensure it's installed and available.")
            pass

    def write_kvs(self, kvs):
        data = kvs.copy()
        step = data.pop("step", None)
        if self.writer:
            for k, v in data.items():
                self.writer.add_scalar(f"logger/{k}", v, step)

    def close(self):
        self.writer.close()


class WandBOutputFormat(KVWriter):
    def __init__(self):
        self.wandb = None
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            error("Unable to import wandb. Please ensure it's installed and available.")
            pass

    def write_kvs(self, kvs):
        data = kvs.copy()
        step = data.pop("step", None)
        if self.wandb and self.wandb.run is not None:
            self.wandb.log(kvs, step=step)


class Logger(object):
    DEFAULT = None
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
    pylogger = logging.getLogger()
    pylogger.setLevel(logging.DEBUG)
    redirect_handler = RedirectLoggingHandler()
    for h in pylogger.handlers:
        if isinstance(h, type(redirect_handler)):
            # handler of this type already exists, prevent duplicating redirects
            return
    pylogger.addHandler(redirect_handler)
    pylogger.info("redirecting to custom logger")


def configure(output_dir, log_suffix="", overwrite=False):
    format_refs = os.getenv("LOGGING_FORMATS", "stdout,log,csv,wandb").split(",")
    output_formats = [get_output_format(f, output_dir, log_suffix, overwrite) for f in format_refs]

    Logger.CURRENT = Logger(output_dir=output_dir, output_formats=output_formats)
    log(f"logging to {output_dir}")
    configure_pylogging_handler()


def get_output_format(format_ref, output_dir, log_suffix="", overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    if format_ref == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format_ref == "log":
        return HumanOutputFormat(osp.join(output_dir, "log%s.txt" % log_suffix), overwrite=overwrite)
    elif format_ref == "json":
        return JSONOutputFormat(osp.join(output_dir, "progress%s.json" % log_suffix), overwrite=overwrite)
    elif format_ref == "csv":
        return CSVOutputFormat(osp.join(output_dir, "progress%s.csv" % log_suffix), overwrite=overwrite)
    elif format_ref == "tensorboard":
        return TensorBoardOutputFormat(osp.join(output_dir, "tb%s" % log_suffix))
    elif format_ref == "wandb":
        return WandBOutputFormat()
    else:
        raise ValueError(f"Unknown format specified: {format_ref}")


def _configure_default_logger():
    output_formats = [HumanOutputFormat(sys.stdout)]
    Logger.CURRENT = Logger(output_dir=None, output_formats=output_formats)
    Logger.DEFAULT = Logger.CURRENT
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
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT
