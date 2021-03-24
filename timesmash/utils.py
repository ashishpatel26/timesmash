import json
import time
import csv
import warnings
import tempfile
import subprocess as sp
import numpy as np
import pandas as pd
import sys
import shutil
import uuid
import os
import atexit
import traceback
import glob
import signal
from functools import partial
import faulthandler, inspect

BIN_PATH = os.path.dirname(os.path.realpath(__file__)) + "/bin/"


def os_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def run_once_per_process(f):
    def wrapper(*args, **kwargs):
        if os.getpgrp() != wrapper.has_run:
            wrapper.has_run = os.getpgrp()
            return f(*args, **kwargs)

    wrapper.has_run = 0
    return wrapper


def process_train_labels(x, y):

    train = pd.DataFrame(x)
    if y is None:
        return train, y
    label = pd.DataFrame(y)
    if label.shape[1] == train.shape[0]:
        label = label.T

    assert label.shape[0] == train.shape[0], "Train and label size mismatch"
    return train, label


def _clean_up_temp_folder(path, signal=None, frame=None):
    if signal is not None:
        traceback.print_stack(frame)
    try:
        [
            os_remove(x) if os.path.isfile(x) else shutil.rmtree(x)
            for x in glob.glob(path + "*")
        ]
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass

    # print(frame.print_stack())
    # aulthandler.dump_traceback()
    # print(traceback)

def getValidKwargs(func, argsDict):

    args = set(inspect.getfullargspec(func).args)
    kwargs = set(inspect.getfullargspec(func).kwonlyargs)
    validKwargs = args.union(kwargs) 
 
    validKwargs.discard('self')
    return dict((key, value) for key, value in argsDict.items() 
                if key in validKwargs)

def callwithValidKwargs(func, argsDict):
    return func(**getValidKwargs(func, argsDict))

@run_once_per_process
def add_signal(path):
    atexit.register(_clean_up_temp_folder, path)
    # signal.signal(signal.SIGINT, partial(_clean_up_temp_folder, path))
    # signal.signal(signal.SIGTERM, partial(_clean_up_temp_folder, path))
    # faulthandler.register(signal.SIGINT)


def RANDOM_NAME(clean=True, path = "ts_temp"):
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, "clean_")
    if clean:
        path = path + str(os.getpgrp()) + "_"
        add_signal(path)
    random_name = str(uuid.uuid4())
    full = path + random_name
    while os.path.isfile(full):
        print("name_double")
        random_name = str(uuid.uuid4())
        full = path + random_name

    return full


class Binary_crashed(Exception):
    def __init__(self, message="Binary crashed!"):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def genesess(
    data_file,
    *,
    outfile=None,
    multi_line=False,
    outfile_suffix="_",
    runfile=False,
    data_type="symbolic",
    data_direction="row",
    gen_epsilon=0.02,
    timer=False,
    data_length=1000000,
    num_steps=20000,
    num_models=1,
    featurization=True,
    depth=1000,
    verbose=False,
    bin_path=BIN_PATH,
    clean=True
):
    """
    """

    if outfile is None:
        outfile = RANDOM_NAME(clean)
    if not isinstance(data_file, str):
        name_file = RANDOM_NAME(clean)
        data_file.to_csv(name_file, sep=" ", index=False, header=False)
        data_file_index = data_file.index
        data_file = name_file
    if not multi_line:
        gen_binary = [os.path.abspath(os.path.join(bin_path, "genESeSS"))]
    else:
        gen_binary = [os.path.abspath(os.path.join(bin_path, "genESeSS_feature"))]

    _data_file = ["-f", data_file]

    if runfile:
        _outfile = ["-R"]
        suffix = "_runfile"
        _num_steps = ["-r", str(num_steps)]
    else:
        _outfile = ["-S"]
        suffix = "_features"
        _num_steps = []
    if outfile is None:
        _outfile.append(data_file + suffix)
    else:
        _outfile.append(outfile)
    output = _outfile[1]

    _data_type = ["-T", data_type]

    _data_direction = ["-D", data_direction]

    _gen_epsilon = ["-e", str(gen_epsilon)]

    _timer = ["-t", str(timer).lower()]

    _data_length = ["-x", str(data_length)]

    _num_models = ["-N", str(num_models)]

    if featurization:
        _featurization = ["-y", "on"]
    else:
        _featurizaiton = []

    _depth = ["-W", str(depth)]

    force_direction = ["-F"]

    _verbose = ["-v"]
    if verbose:
        _verbose.append("1")
    else:
        _verbose.append("0")

    command_list = (
        gen_binary
        + _data_file
        + _outfile
        + _data_type
        + _data_direction
        + _gen_epsilon
        + _timer
        + _data_length
        + _num_steps
        + _num_models
        + _featurization
        + _depth
        + force_direction
        + _verbose
    )
    # print(' '.join(command_list))
    # print(output, name_file)

    sp.run(command_list, encoding="utf-8")
    try:
        data = pd.read_csv(output, header=None, sep=" ").dropna(how="all", axis=1)
    except:
        data = pd.DataFrame(
            np.zeros((data_file_index.shape[0], depth)),
            index=data_file_index,
            columns=[x for x in range(depth)],
        )
    if data.empty:
        data = pd.DataFrame(
            np.zeros((data_file_index.shape[0], depth)),
            index=data_file_index,
            columns=[x for x in range(depth)],
        )
    if clean:
        os_remove(name_file)
        os_remove(output)
        # data = data.loc[:, (data != 0).any(axis=0)]
    # print(data)
    # sys.exit()
    return data


def xgenesess(
    data_file,
    *,
    outfile=None,
    data_type="symbolic",
    num_lines="all",
    partition=None,
    detrending=None,
    min_delay=0,
    max_delay=30,
    bin_path=BIN_PATH,
    clean=True
):
    """
    """
    outfile = RANDOM_NAME(clean) if outfile is None else outfile
    is_file = False
    if not isinstance(data_file, str):
        name_file = RANDOM_NAME(clean)
        data_file.to_csv(name_file, sep=" ", index=False, header=False)
        is_file = True
        data_file = name_file
    xgen_binary = [os.path.abspath(os.path.join(bin_path, "XgenESeSS"))]

    _data_file = ["-f", data_file]

    _data_type = ["-T", data_type]

    _outfile = ["-Y"]
    if outfile is None:
        _outfile.append(data_file + "_")
    else:
        _outfile.append(outfile)

    if num_lines == "all":
        with open(data_file) as infile:
            _num_lines = sum(1 for _ in infile)
        _num_lines = num_lines_arg(_num_lines)
    # elif num_lines == 'one':
    else:
        _num_lines = "'0:0'"
    _selector = ["-k", _num_lines]

    if partition is None:
        _partition = []
    elif isinstance(partition, int):
        _partition = ["-p", str(partition)]
    elif isinstance(partition, list):
        _partition = ["-p"] + [str(p) for p in partition]

    if detrending is None:
        _detrending = []
    else:
        _detrending = ["-u", str(detrending)]

    _min_delay = ["-B", str(min_delay)]
    _max_delay = ["-E", str(max_delay)]

    _infer_model = ["-S"]
    _print_gamma = ["-y", "1"]
    _no_loading = ["-q"]

    command_list = (
        xgen_binary
        + _data_file
        + _selector
        + _data_type
        + _partition
        + _detrending
        + _infer_model
        + _min_delay
        + _max_delay
        + _print_gamma
        + _outfile
        + _no_loading
    )

    command_list = " ".join(command_list)
    try:
        sp.run(command_list, shell=True)
    except:
        raise Binary_crashed()
    if is_file:
        data = (
            pd.read_csv(_outfile[1], header=None, sep=" ")
            .dropna(how="all")
            .dropna(how="all", axis=1)
        )
        if clean:
            os_remove(_outfile[1])
            os_remove(name_file)
        return data
    return _outfile[1]


def serializer(
    bmp_filenames,
    *,
    outfile,
    bin_path=BIN_PATH,
    seq_len=1000,
    num_seqs=1,
    power_coeff=1.0,
    channel="R",
    size=16384,
    serializer_verbose=False
):

    serializer_binary = [os.path.abspath(os.path.join(bin_path, "serializer"))]
    _bmp_filenames = ["-f", bmp_filenames]
    _outfile = ["-o", outfile]
    _seq_len = ["-L", str(seq_len)]
    _num_seqs = ["-n", str(num_seqs)]
    _power_coeff = ["-w", str(power_coeff)]
    _channel = ["-c", channel]
    _size = ["-s", str(size)]
    _verbose = ["-v"]
    if serializer_verbose:
        _verbose.append("1")
    else:
        _verbose.append("0")

    command_list = (
        serializer_binary
        + _bmp_filenames
        + _outfile
        + _seq_len
        + _num_seqs
        + _power_coeff
        + _channel
        + _size
        + _verbose
    )
    return sp.check_output(command_list, encoding="utf-8")


def smash(
    data_file,
    *,
    outfile=None,
    partition=None,
    data_type="symbolic",
    data_direction="row",
    num_reruns=20,
    bin_path=BIN_PATH
):
    """
    """
    outfile = RANDOM_NAME(clean) if outfile is None else outfile
    is_file = False
    if not isinstance(data_file, str):
        file_name = RANDOM_NAME(clean)
        data_file.to_csv(file_name, sep=" ", index=False, header=False)
        is_file = True
        data_file = file_name

    smash_binary = [os.path.abspath(os.path.join(bin_path, "smash"))]
    _data_file = ["-f", data_file]
    _outfile = ["-o", outfile]

    if partition is None:
        _partition = []
    elif type(partition) is int:
        _partition = ["-p", str(partition)]
    elif type(partition) is list:
        _partition = ["-p"] + [str(p) for p in partition]

    _data_type = ["-T", data_type]

    _data_direction = ["-D", data_direction]

    _num_reruns = ["-n", str(num_reruns)]

    command_list = (
        smash_binary
        + _data_file
        + _outfile
        + _partition
        + _data_type
        + _data_direction
        + _num_reruns
    )

    try:
        sp.check_output(command_list)
    except:
        raise Binary_crashed()

    results = np.loadtxt(outfile, dtype=float)
    if is_file:
        os_remove(data_file)
        os_remove(outfile)
    return results


def smashmatch(
    data_file,
    *,
    lib_files,
    output_prefix,
    partition=None,
    data_type="symbolic",
    data_direction="row",
    num_reruns=20,
    bin_path=BIN_PATH
):
    """
    """
    smash_binary = [os.path.abspath(os.path.join(bin_path, "smashmatch"))]
    _data_file = ["-f", data_file]
    _lib_files = ["-F"] + lib_files

    _outfile = ["-o", output_prefix]

    if partition is None:
        _partition = []
    elif type(partition) is int:
        _partition = ["-p", str(partition)]
    elif type(partition) is list:
        _partition = ["-p"] + [str(p) for p in partition]

    _data_type = ["-T", data_type]

    _data_direction = ["-D", data_direction]

    _num_reruns = ["-n", str(num_reruns)]

    command_list = (
        smash_binary
        + _data_file
        + _outfile
        + _lib_files
        + _partition
        + _data_type
        + _data_direction
        + _num_reruns
    )
    try:
        sp.check_output(command_list)
    except sp.CalledProcessError as e:
        raise Binary_crashed()

    probs = pd.read_csv(output_prefix + "_prob", header=None, sep=" ")
    if probs.dropna(axis=1).shape[1] == 1:
        probs[1] = 0
        probs.to_csv(output_prefix + "_prob", sep=" ", header=False, index=False)


def argmax_prod_matrix_list(matrix_list, *, index_class_map, axis=1):
    """
    """
    start = np.ones(matrix_list[0].shape)
    for matrix in matrix_list:
        start *= matrix
    argmaxes = np.argmax(start, axis=axis)
    if index_class_map:
        predictions = []
        for i in argmaxes:
            predictions.append(index_class_map[i])
        return predictions
    else:
        return argmaxes


def num_lines_arg(n):
    """
    """
    ratio_list = []
    for i in range(n):
        next_ratio = str(i) + ":" + str(i)
        ratio_list.append(next_ratio)
    ratio_string = " | ".join(ratio_list)
    ratio_argument = str(repr(ratio_string))
    return ratio_argument


def predict_random(class_list, test_file):
    """
    """
    with open(test_file, "r") as infile:
        num_predictions = sum(1 for _ in infile)
    random_predictions = np.random.choice(class_list, size=num_predictions)
    return random_predictions


def _lsmash(data, *, data_file=None, distance_file=None, clean=True):
    """
    """
    data_file = RANDOM_NAME(clean) if data_file is None else data_file
    distance_file = RANDOM_NAME(clean) if distance_file is None else distance_file
    data.to_csv(data_file, sep=" ", index=False, header=False)
    out = sp.check_output(
        BIN_PATH
        + "lsmash -f "
        + data_file
        + " -u 1 -S 0 -D row "
        + " -T symbolic -o "
        + distance_file,
        shell=True,
    )
    dist = pd.read_csv(distance_file, sep=" ", header=None).dropna(how="all", axis=1)
    if clean:
        os_remove(data_file)
        os_remove(distance_file)
    dist.index = data.index
    dist.columns = data.index
    return dist


def _gen_model(data, *, eps=0.01, clean=True):
    out_file = RANDOM_NAME(clean)
    data_file = RANDOM_NAME(clean)
    data.to_csv(data_file, sep=" ", index=False, header=False)
    gen_binary = os.path.abspath(os.path.join(BIN_PATH, "genESeSS"))
    call = (
        gen_binary
        + " -f "
        + data_file
        + " -t off"
        + " -F -v 0 -D row -T symbolic -o "
        + out_file
        + " -e "
        + str(eps)
    )
    out = sp.check_output(call, shell=True)
    if clean:
        os_remove(data_file)
    return out_file


def _llk_state(data, model, *, clean=True, data_file=None):
    data_file = RANDOM_NAME(clean) if data_file is None else data_file
    data.to_csv(data_file, sep=" ", index=False, header=False)
    llk_binary = os.path.abspath(os.path.join(BIN_PATH, "llk_state"))
    llpos = llk_binary + " -s " + data_file + " -f " + model
    try:
        bin_ret = sp.check_output(llpos, shell=True).decode("ascii").strip("\n")
    except sp.CalledProcessError as e:
        os_remove(data_file)
        raise Binary_crashed()
    bin_ret = [x.strip().split() for x in bin_ret.split("\n")]
    df_toreturn = pd.DataFrame(bin_ret).astype(float)
    if clean:
        os_remove(data_file)

    df_toreturn.index = data.index
    return df_toreturn


from contextlib import contextmanager


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def _llk(data, model, *, clean=True, data_file=None):
    data_file = RANDOM_NAME(clean) if data_file is None else data_file
    data.to_csv(data_file, sep=" ", index=False, header=False)
    llk_binary = os.path.abspath(os.path.join(BIN_PATH, "llk"))
    llpos = llk_binary + " -s " + data_file + " -f " + model
    if True:
        try:
            df_toreturn = pd.DataFrame(
                sp.check_output(llpos, shell=True).split()
            ).astype(float)

        except sp.CalledProcessError as e:
            if clean:
                os_remove(data_file)
            raise Binary_crashed()
    if clean:
        os_remove(data_file)

    df_toreturn.index = data.index
    return df_toreturn
