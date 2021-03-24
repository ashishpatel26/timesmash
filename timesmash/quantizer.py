"""
Quantizer class and functions.
"""
import os
import csv
import sys
from time import time
import tempfile
import subprocess as sp
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
from timesmash.utils import RANDOM_NAME, BIN_PATH, process_train_labels


class Quantizer(object):
    """
    Quantizer class that is used to quantize continuous 
    data. 
    @author zed.uchicago.edu
    Inputs:
        sample_size (float): percent of train data to use for fitting
        bin_path (string): path to quantizer binary
        epsilon (int): -1 for not grouping the data for 
                                partition, a positive value gives the 
                                resolution for grouping.
        min_alphabet_size (int): The smallest alphabet size for grid search.
        max_alphabet_size (int): The biggest alphabet size for grid search.
        detrending (int): The number of times the raw data is 
                                differentiated. Choose between 0 and 1 
                                if not specified
        normalization (int): Whether to normalized the data. Try both
                                options if not specified
        n_quantizations (int or 'max'): the number of best parameters to output
        return_failed (bool): returns quantizations with one symbool in train
        verbose (bool): prints debugging output if true 
        clean (bool): removes temporary files if true
    """

    def __init__(
        self,
        *,
        sample_size=1,
        bin_path=BIN_PATH,
        pooled=True,
        epsilon=-1,
        min_alphabet_size=2,
        max_alphabet_size=3,
        detrending=None,
        normalization=None,
        n_quantizations="max",
        return_failed=True,
        prune=False,
        verbose=False,
        clean=True
    ):

        self._is_supervised = True
        self._return_failed = return_failed
        self._bin_path = bin_path
        self._pooled = pooled
        self._epsilon = epsilon
        self._min_alphabet_size = min_alphabet_size
        self._max_alphabet_size = max_alphabet_size
        self._detrending = detrending
        self._normalization = normalization
        self._num_quantizations = n_quantizations
        self._verbose = verbose
        self._command_list = []
        self._fitted = False
        self._clean = clean
        self._prune = prune
        self._feature_order = []
        self._sample_size = sample_size
        self.quantized_data_dir = tempfile.mkdtemp(prefix="quantized_data_")

        self._data_dir = RANDOM_NAME(clean=clean)
        self.parameters = {}
        self.data = []
        self._partition_success_set = set()
        self.training_X = None
        self.parameter_index_map = {}

    def fit(self, data_dir, label=None, *, force_refit=False):
        if self._num_quantizations == 0:
            return data_dir
        if self._fitted and not force_refit:
            return None
        if (label is None) and (type(data_dir) is not str):
            self.is_supervised = False
            label = pd.DataFrame([1] * data_dir.shape[0])
        if type(data_dir) is not str:
            data_dir, label = process_train_labels(data_dir, label)
            if os.path.exists(self._data_dir):
                shutil.rmtree(self._data_dir)
            os.mkdir(self._data_dir)
            file = open(os.path.join(self._data_dir, "library_list"), "w")
            classes = set(label[list(label)[0]])
            for i in classes:
                outfile = "train_class_" + str(i)
                #print(data_dir, i)
                label1 = label.T


                df_i = data_dir.loc[label[label[list(label)[0]] == i].index]
                df_i.to_csv(
                    os.path.join(self._data_dir, outfile),
                    sep=" ",
                    index=False,
                    header=False,
                )
                file.write(
                    outfile + " " + str(i) + " " + str(df_i.shape[0]) + " " + "\n"
                )
            file.close()
            self.data_dir = self._data_dir
        else:
            self.data_dir = data_dir

        size = data_dir.shape[0] * data_dir.shape[1]
        self._sample_size = min(1, 100000 / size)
        self._get_command_list(self.data_dir)
        raw_output = sp.check_output(self._command_list, encoding="utf-8")

        # if self._problem_type == 'supervised':
        self._note_lib_files(self.data_dir)

        prune_range_list = []
        detrending_list = []
        normalization_list = []
        partition_list = []

        valid_params_path = os.path.join(self.data_dir, "valid_parameter")
        with open(valid_params_path) as f:
            parameters = f.read().splitlines()
        if self._clean:
            shutil.rmtree(self._data_dir)
        parameters.sort()  # sorts normally by alphabetical order
        parameters.sort(key=len, reverse=True)  # sorts by descending length
        for param in parameters:
            pr, d, n, pa = self._read_quantizer_params(param)
            prune_range_list.append(pr)
            detrending_list.append(d)
            normalization_list.append(n)
            partition_list.append(pa)

        parameter_zip = zip(
            prune_range_list, detrending_list, normalization_list, partition_list
        )
        parameters = {}
        for pr, d, n, pa in parameter_zip:
            key = self._write_quantizer_params(pr, d, n, pa)
            param_set = {
                "prune_range": pr,
                "detrending": d,
                "normalization": n,
                "partition": pa,
            }
            parameters[key] = param_set

            # don't include duplicate quantizations
            if key not in self._feature_order:
                self._feature_order.append(key)
        assert (
            len(self._feature_order) != 0
        ), "Quantization failed, try manual quantization or normalize the data."
        self.parameters = parameters
        self._fitted = True
        return self

    def transform(self, data, *, output_type="matrix"):
        if self._num_quantizations == 0:
            return data
        assert self._fitted, (
            "'fit()' or 'fit_transform()' must be called"
            + " prior to running 'transform()'"
        )
        ""
        if isinstance(data, pd.DataFrame):
            """
            if data.nunique(axis=1).max() < 4:
                return pd.concat[data*len(self._feature_order)]
            """
            data_name = "random"

        else:
            data_name = os.path.basename(data)
            if not os.path.isdir(self.data_dir):
                os.mkdir(self.data_dir)
            if not os.path.isdir(self.quantized_data_dir):
                os.mkdir(self.quantized_data_dir)
        data_prefix = os.path.basename(data_name) + "_"
        # qdata_dir = tempfile.mkdtemp(prefix=data_prefix, dir=self.quantized_data_dir)
        qdata_dir = ""
        data_dict = {}
        data_dict[data_name] = {}
        data_dict[data_name]["files"] = {}
        # data_dict[data_name]["directory"] = qdata_dir

        # TODO: TESTING
        partition_total = len(self.parameters)
        partition_fail_num = 0
        # TODO: END TESTING
        success = []
        # print(self.parameters)
        for i, name in enumerate(self._feature_order):
            # for name, p_dict in self.parameters.items():

            p_dict = self.parameters[name]
            data_with_params = data_prefix + name
            data_with_params_path = os.path.join(qdata_dir, data_with_params)
            res = self._try_apply_quantizer(
                data, outfile=data_with_params_path, **p_dict
            )
            if res is None:
                success.append(res)
            else:
                success.append(True)
            yield res
        if isinstance(data, pd.DataFrame):
            assert not all(
                v is None for v in success
            ), "All quantizations failed on train"

    def fit_transform(
        self, data_dir, label=None, *, output_type="matrix", force_refit=False
    ):

        self.fit(data_dir, label=label, force_refit=force_refit)
        if type(data_dir) is not str:
            return self.transform(data_dir)
        # lib_list = self.data #list(self.data.keys())
        X = []
        for lib_file in self.data:  # lib_list:
            lib_path = os.path.join(data_dir, lib_file)
            X_ = self.transform(lib_path, output_type=output_type)
            if X_ is None:
                return None
            X.append(X_)
        X_ = np.vstack(X)
        self.training_X = X_
        if output_type == "filename":
            X_ = X
        return X_

    def _get_command_list(self, data_dir):

        problem_type = ["-t"]
        num_streams = ["-T"]
        if self._is_supervised:
            problem_type.append("2")
            num_streams.append("-1")

        quantizer_path = os.path.join(self._bin_path, "Quantizer")
        quantizer_binary = [os.path.abspath(quantizer_path)]
        data_dir = ["-D", data_dir]

        sample_size = ["-x", str(self._sample_size)]
        pooled = ["-w"]
        if self._pooled:
            pooled.append("1")
        else:
            pooled.append("0")

        epsilon = ["-e", str(self._epsilon)]

        min_alphabet_size = ["-a", str(self._min_alphabet_size)]
        max_alphabet_size = ["-A", str(self._max_alphabet_size)]

        if self._detrending is not None:
            detrending = ["-d", str(int(self._detrending))]
        else:
            detrending = []

        if self._normalization is not None:
            normalization = ["-n", str(int(self._normalization))]
        else:
            normalization = []

        if self._prune is not None:
            prunning = ["-r", str(int(self._prune))]
        else:
            prunning = []

        if self._num_quantizations == "max":
            num_partitions = []
        else:
            num_partitions = ["-M", str(int(self._num_quantizations))]

        command_list = (
            quantizer_binary
            + data_dir
            + problem_type
            + prunning
            + ["-f", str(int(self._return_failed))]
            # + num_streams # TODO: testing Quantizer_v1
            + sample_size
            # + pooled # TODO: testing Quantizer_v1
            + epsilon
            + min_alphabet_size
            + max_alphabet_size
            + detrending
            + normalization
            + num_partitions
        )
        self._command_list = command_list

    def _note_lib_files(self, data_dir):

        library_list = os.path.join(data_dir, "library_list")
        if os.path.isfile(library_list):
            with open(library_list) as f:
                train_data = [row.split(" ")[:2] for row in f.read().splitlines()]
        else:
            train_data = [["dataset", -1]]
        for lib, label_str in train_data:
            self.data.append(lib)

    @staticmethod
    def _detrend(df, *, detrend_level):

        return df.diff(axis=1).dropna(how="all", axis=1)
    '''
    @staticmethod
    def _normalize(df):
        X_scaled = pd.DataFrame(
            preprocessing.scale(df, axis=1), index=df.index, columns=df.columns
        )

        return X_scaled
    '''
    @staticmethod
    def _normalize(df):
        """
        """
        df_stdev = df.std(axis=1)
        pos_stdev_0 = df_stdev==0
        df_stdev[pos_stdev_0] = 1

        standard_normal_rows = df.subtract(df.mean(axis=1),
                                           axis=0).divide(df_stdev,
                                                          axis=0)
        return standard_normal_rows
    
    
    @staticmethod
    def _prune_func(df, lower_bound, upper_bound):
        """

        """
        for index in df.index:
            X = []
            for val in df.loc[index].values:
                if val <= float(lower_bound) or val >= float(upper_bound):
                    X = np.append(X, val)
            pruned_ = np.empty([1, len(df.loc[index].values) - len(X)])
            pruned_[:] = np.nan
            X = np.append(X, pruned_)
            df.loc[index] = X
        return df

    def _try_apply_quantizer(
        self,
        data,
        *,
        partition,
        prune_range=None,
        detrending=None,
        normalization=None,
        outfile=None,
        verbose=False
    ):
        # data.to_csv("debug0")
        max_col_len = 0
        if type(data) is str:
            with open(data, "r") as infile:
                csv_reader = csv.reader(infile, delimiter=" ")
                for row in csv_reader:
                    len_ = len(row)
                    if len_ > max_col_len:
                        max_col_len = len_
            unquantized = pd.read_csv(
                data,
                delimiter=" ",
                dtype="float",
                header=None,
                names=range(max_col_len),
            )
        else:
            unquantized = data.copy()
        columns_list = list(unquantized.columns)
        index_list = unquantized.index
        if prune_range:
            if verbose:
                print("PRUNING")
            unquantized = self._prune_func(unquantized, prune_range[0], prune_range[1])

        # pd.DataFrame(unquantized).to_csv("debug1")
        if detrending:
            if verbose:
                print("DETRENDING")
            unquantized = self._detrend(unquantized, detrend_level=detrending)
        # pd.DataFrame(unquantized).to_csv("debug2")
        if normalization:
            if verbose:
                print("NORMALIZING")
            unquantized = self._normalize(unquantized)
        # unquantized.to_csv("debug3")
        if outfile is None:
            _outfile = filename
        else:
            _outfile = outfile
        if type(data) is not str:
            partition = [float(i) for i in partition]
            quantized = pd.DataFrame(
                np.digitize(unquantized.values, bins=partition), index=data.index
            )
            # quantized[unquantized.isnull()] = np.nan
            if (
                not self._correct_num_symbols(quantized.values, partition)
                and not self._return_failed
            ):
                print("Dropping invalid quantization.")
                return None
            return quantized

        if not os.path.isfile(_outfile):
            print("Failed to apply quantization! Retrying...")
            self._try_apply_quantizer(
                filename,
                partition=partition,
                prune_range=prune_range,
                detrending=detrending,
                normalization=normalization,
                outfile=outfile,
                verbose=verbose,
            )
        return True

    @staticmethod
    def _correct_num_symbols(quantized_matrix, partition):

        expected_num_symbols = len(partition) + 1
        i = 0
        for row in quantized_matrix:
            num_symbols = len(np.unique(row))
            if num_symbols == 1:
                # print("SINGLE SYMBOL STREAM SOMEHOW PASSED CHECK: {}".format(partition))
                pass
            if num_symbols != expected_num_symbols:
                # np.savetxt("foo.csv", quantized_matrix, delimiter=",")
                return False
            i += 1

        return True

    @staticmethod
    def _read_quantizer_params(parameters):

        parameters_ = parameters.split("L")[0]
        prune_range = []
        for index, char in enumerate(parameters_):
            if char == "R":
                for char_ in parameters_[index + 2 :]:
                    if char_ != "]":
                        prune_range.append(char_)
                    else:
                        break
            elif char == "D":
                detrending = int(parameters_[index + 1])
            elif char == "N":
                normalization = int(parameters_[index + 1])
        if prune_range:
            prune_range = "".join(prune_range).split(" ")

        partition = parameters_.split("[")[-1].strip("]").split()
        no_negative_zero_partition = []
        for p in partition:
            if repr(float(p)) == "-0.0":
                p_ = p[1:]
            else:
                p_ = p
            no_negative_zero_partition.append(p_)

        return prune_range, detrending, normalization, no_negative_zero_partition

    @staticmethod
    def _write_quantizer_params(prune_range, detrending, normalization, partition):
        """

        """
        params = []
        if prune_range:
            params.append("R")
            params += prune_range
        if detrending:
            params.append("D")
            params.append(detrending)
        if normalization:
            params.append("N")
            params.append(normalization)
        if partition:
            params.append("P")
            params += str([float(p) for p in partition]).replace(" ", "")
        params_string = "".join([str(p) for p in params])
        return params_string

    @staticmethod
    def _get_num_cols(infile):
        with open(infile) as f:
            first_line = f.readline()
        first_line_list = first_line.strip.split(" ")
        num_col = len(first_line_list)
        return num_col

    def get_n_quantizations(self):
        return len(self._feature_order)

    def set_quantization(self, pa, *, pr=[], d=0, n=0):
        key = self._write_quantizer_params(pr, d, n, pa)
        param_set = {
            "prune_range": pr,
            "detrending": d,
            "normalization": n,
            "partition": pa,
        }
        self.parameters[key] = param_set
        # don't include duplicate quantizations
        if key not in self._feature_order:
            self._feature_order.append(key)
        self._fitted = True
