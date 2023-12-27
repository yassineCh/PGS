import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb

import torch
import design_bench

def get_task_and_dataset(task_name, map_to_logits=False):
    """This method returns the offline dataset and the task object 
    for a given task. The task object is required to call the 
    oracle for evaluating candidate designs.
    """
    if task_name == "tfbind8":
        task = design_bench.make("TFBind8-Exact-v0")
        from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
        task_dataset = TFBind8Dataset()
    elif task_name == "gfp":
        task = design_bench.make("GFP-Transformer-v0")
        from design_bench.datasets.discrete.gfp_dataset import GFPDataset
        task_dataset = GFPDataset()
    elif task_name == "utr":
        task = design_bench.make("UTR-ResNet-v0")
        from design_bench.datasets.discrete.utr_dataset import UTRDataset
        task_dataset = UTRDataset()
    elif task_name == "superconductor":
        task = design_bench.make(f"Superconductor-RandomForest-v0")
        from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
        task_dataset = SuperconductorDataset()
    elif task_name == "ant":
        task = design_bench.make(f"AntMorphology-Exact-v0")
        from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
        task_dataset = AntMorphologyDataset()
    elif task_name == "dkitty":
        task = design_bench.make(f"DKittyMorphology-Exact-v0")
        from design_bench.datasets.continuous.dkitty_morphology_dataset import  DKittyMorphologyDataset
        task_dataset = DKittyMorphologyDataset()

    if task_name in ["tfbind8", "gfp", "utr"] and map_to_logits is True:
        task.map_to_logits()
        task_dataset.map_to_logits()
    
    return task, task_dataset

class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = 'SimpleSAC'
        config.project = 'sac'
        config.output_dir = '/tmp/SimpleSAC'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output



def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }
