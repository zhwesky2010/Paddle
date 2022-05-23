#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.common_ops_import import dygraph_only
from paddle import _C_ops

__all__ = []


@dygraph_only
def mm(x, y):
    '''

    '''
    return _C_ops.final_state_sparse_mm(x, y)


@dygraph_only
def mm_mask(x, y, mask):
    '''

    '''
    return _C_ops.final_state_sparse_mm_mask(x, y)
