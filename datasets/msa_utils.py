# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shutil
import string
import subprocess
import time
import uuid
from collections import OrderedDict, defaultdict
from os.path import exists as opexists
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np


PROT_TYPE_NAME = "proteinChain"  # inference protein name in json


# FeatureDict, make_dummy_msa_obj, convert_monomer_features
# These are modified from openfold: data/data_pipeline
try:
    from openfold_local.data.tools import jackhmmer
except ImportError:
    print(
        "Failed to import packages for searching MSA; can only run with precomputed MSA"
    )

logger = logging.getLogger(__name__)
FeatureDict = MutableMapping[str, np.ndarray]