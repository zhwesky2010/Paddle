# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
When training a model, it's often useful to decay the
learning rate during training process, this is called
learning_rate_decay. There are many strategies to do
this, this module will provide some classical method.
User can also implement their own learning_rate_decay
strategy according to this module.
"""

from __future__ import print_function

import math
import numbers

from . import control_flow
from . import nn
from . import ops
from . import tensor
from ..framework import default_main_program, Parameter, unique_name, name_scope
from ..framework import Variable
from ..framework import in_dygraph_mode
from ..dygraph import learning_rate_scheduler as imperate_lr
from ..data_feeder import check_variable_and_dtype, check_type

__all__ = [
    'exponential_decay', 'natural_exp_decay', 'inverse_time_decay',
    'polynomial_decay', 'piecewise_decay', 'noam_decay', 'cosine_decay',
    'linear_lr_warmup', 'reduce_lr_on_plateau'
]


def _decay_step_counter(begin=0):
    # the first global step is zero in learning rate decay
    global_step = nn.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    global_step = tensor.cast(global_step, 'float32')
    return global_step


def noam_decay(d_model, warmup_steps, learning_rate=1.0):
    """
    Noam decay method. The numpy implementation of noam decay as follows.

    .. code-block:: python
      
      import paddle.fluid as fluid
      import numpy as np
      # set hyper parameters
      base_lr = 0.01
      d_model = 2
      current_steps = 20
      warmup_steps = 200
      # compute
      lr_value = base_lr * np.power(d_model, -0.5) * np.min([
                              np.power(current_steps, -0.5),
                              np.power(warmup_steps, -1.5) * current_steps])

    Please reference `attention is all you need
    <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        d_model(Variable): The dimensionality of input and output of model.

        warmup_steps(Variable): A super parameter.

        learning_rate(Variable|float|int): The initial learning rate. If the type
            is Variable, it's a tensor with shape [1], the data type can be
            float32 or float64. It also can be set to python int number. Default 1.0

    Returns:
        The decayed learning rate.
    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          warmup_steps = 100
          learning_rate = 0.01
          lr = fluid.layers.learning_rate_scheduler.noam_decay(
                         1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps,
                         learning_rate)
    """
    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.NoamDecay(
                d_model, warmup_steps, learning_rate=learning_rate)
            return decay
        else:
            global_step = _decay_step_counter(1)

            a = global_step**-0.5
            b = (warmup_steps**-1.5) * global_step
            lr_value = learning_rate * (d_model**-0.5) * nn.elementwise_min(a,
                                                                            b)

            return lr_value


def exponential_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """
    Applies exponential decay to the learning rate.

    When training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, the learning rate will be decayed by
    'decay_rate' every 'decay_steps' steps.

    Decayed learning rate calculates as follows:

    >>> if staircase == True:
    >>>     decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    >>> else:
    >>>     decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

    Args:
        learning_rate(Variable|float): The initial learning rate. It should be a Variable 
                                       or a float
        decay_steps(int): The learning rate decay steps. See the decay computation above.
        decay_rate(float): The learning rate decay rate. See the decay computation above.
        staircase(bool): If True, decay the learning rate at discrete intervals, which 
                         means the learning rate will be decayed by `decay_rate` every
                         `decay_steps`. If False, learning rate will be decayed continuously
                         and following the formula above. Default: False

    Returns:
        Variable: The decayed learning rate. The data type is float32.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          sgd_optimizer = fluid.optimizer.SGD(
	      learning_rate=fluid.layers.exponential_decay(
		    learning_rate=base_lr,
		    decay_steps=10000,
		    decay_rate=0.5,
		    staircase=True))

    """
    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.ExponentialDecay(learning_rate, decay_steps,
                                                 decay_rate, staircase)
            return decay
        else:
            global_step = _decay_step_counter()

            div_res = global_step / decay_steps
            if staircase:
                div_res = ops.floor(div_res)
            decayed_lr = learning_rate * (decay_rate**div_res)

            return decayed_lr


def natural_exp_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """Applies natural exponential decay to the initial learning rate.

    When training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, the learning rate will be decayed by
    natural exponential power 'decay_rate' every 'decay_steps' steps.

    Decayed learning rate calculates as follows:

    >>> if not staircase:
    >>>     decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    >>> else:
    >>>     decayed_learning_rate = learning_rate * exp(- decay_rate * floor(global_step / decay_steps))

    Args:
        learning_rate(Variable|float): The initial learning rate. It should be a Variable 
                                       or a float
        decay_steps(int): The learning rate decay steps. See the decay computation above.
        decay_rate(float): The learning rate decay rate. See the decay computation above.
        staircase(bool): If True, decay the learning rate at discrete intervals, which 
                         means the learning rate will be decayed by natural exponential power
                         `decay_rate` every `decay_steps`. If False, learning rate will be
                         decayed continuously and following the formula above. Default: False

    Returns:
        The decayed learning rate. The data type is float32.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          sgd_optimizer = fluid.optimizer.SGD(
	      learning_rate=fluid.layers.natural_exp_decay(
		    learning_rate=base_lr,
		    decay_steps=10000,
		    decay_rate=0.5,
		    staircase=True))

    """
    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.NaturalExpDecay(learning_rate, decay_steps,
                                                decay_rate, staircase)
            return decay
        else:
            global_step = _decay_step_counter()

            div_res = global_step / decay_steps
            if staircase:
                div_res = ops.floor(div_res)
            decayed_lr = learning_rate * ops.exp(-1 * decay_rate * div_res)

            return decayed_lr


def inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """
    Applies inverse time decay to the initial learning rate.

    When training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, an inverse decay function will be
    applied to the initial learning rate.

    Decayed learning rate calculates as follows:

    >>> if staircase == True:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    >>> else:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

    Args:
        learning_rate(Variable|float): The initial learning rate. It should be a Variable 
                                       or a float
        decay_steps(int): The learning rate decay steps. See the decay computation above.
        decay_rate(float): The learning rate decay rate. See the decay computation above.
        staircase(bool): If True, decay the learning rate at discrete intervals, which 
                         means the learning rate will be decayed by `decay_rate` times 
                         every `decay_steps`. If False, learning rate will be decayed 
                         continuously and following the formula above. Default: False

    Returns:
        Variable: The decayed learning rate. The data type is float32.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          sgd_optimizer = fluid.optimizer.SGD(
	      learning_rate=fluid.layers.inverse_time_decay(
		    learning_rate=base_lr,
		    decay_steps=10000,
		    decay_rate=0.5,
		    staircase=True))
    """
    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.InverseTimeDecay(learning_rate, decay_steps,
                                                 decay_rate, staircase)
            return decay
        else:
            global_step = _decay_step_counter()

            div_res = global_step / decay_steps
            if staircase:
                div_res = ops.floor(div_res)

            decayed_lr = learning_rate / (1 + decay_rate * div_res)

            return decayed_lr


def polynomial_decay(learning_rate,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    """
    Applies polynomial decay to the initial learning rate.

    .. code-block:: text

     if cycle:
       decay_steps = decay_steps * ceil(global_step / decay_steps)
     else:
       global_step = min(global_step, decay_steps)
       decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

    Args:
        learning_rate(Variable|float32): A scalar float32 value or a Variable. This
          will be the initial learning rate during training.
        decay_steps(int32): A Python `int32` number.
        end_learning_rate(float): A Python `float` number.
        power(float): A Python `float` number.
        cycle(bool): If set true, decay the learning rate every decay_steps.

    Returns:
        Variable: The decayed learning rate

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          start_lr = 0.01
          total_step = 5000
          end_lr = 0
          lr = fluid.layers.polynomial_decay(
              start_lr, total_step, end_lr, power=1)

    """
    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.PolynomialDecay(learning_rate, decay_steps,
                                                end_learning_rate, power, cycle)
            return decay
        else:
            global_step = _decay_step_counter()

            if cycle:
                div_res = ops.ceil(global_step / decay_steps)
                zero_var = tensor.fill_constant(
                    shape=[1], dtype='float32', value=0.0)
                one_var = tensor.fill_constant(
                    shape=[1], dtype='float32', value=1.0)

                with control_flow.Switch() as switch:
                    with switch.case(global_step == zero_var):
                        tensor.assign(input=one_var, output=div_res)
                decay_steps = decay_steps * div_res
            else:
                decay_steps_var = tensor.fill_constant(
                    shape=[1], dtype='float32', value=float(decay_steps))
                global_step = nn.elementwise_min(
                    x=global_step, y=decay_steps_var)

            decayed_lr = (learning_rate - end_learning_rate) * \
                ((1 - global_step / decay_steps) ** power) + end_learning_rate
            return decayed_lr


def piecewise_decay(boundaries, values):
    """Applies piecewise decay to the initial learning rate.

    The algorithm can be described as the code below.

    .. code-block:: text

      boundaries = [10000, 20000]
      values = [1.0, 0.5, 0.1]
      if step < 10000:
          learning_rate = 1.0
      elif 10000 <= step < 20000:
          learning_rate = 0.5
      else:
          learning_rate = 0.1
    Args:
        boundaries: A list of steps numbers.
        values: A list of learning rate values that will be picked during
            different step boundaries.

    Returns:
        The decayed learning rate.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          optimizer = fluid.optimizer.Momentum(
              momentum=0.9,
              learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
              regularization=fluid.regularizer.L2Decay(1e-4))


    """
    with default_main_program()._lr_schedule_guard():
        if len(values) - len(boundaries) != 1:
            raise ValueError("len(values) - len(boundaries) should be 1")

        if in_dygraph_mode():
            decay = imperate_lr.PiecewiseDecay(boundaries, values, 0)
            return decay
        else:
            global_step = _decay_step_counter()

            lr = tensor.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="learning_rate")

            with control_flow.Switch() as switch:
                for i in range(len(boundaries)):
                    boundary_val = tensor.fill_constant(
                        shape=[1],
                        dtype='float32',
                        value=float(boundaries[i]),
                        force_cpu=True)
                    value_var = tensor.fill_constant(
                        shape=[1], dtype='float32', value=float(values[i]))
                    with switch.case(global_step < boundary_val):
                        tensor.assign(value_var, lr)
                last_value_var = tensor.fill_constant(
                    shape=[1],
                    dtype='float32',
                    value=float(values[len(values) - 1]))
                with switch.default():
                    tensor.assign(last_value_var, lr)

            return lr


def cosine_decay(learning_rate, step_each_epoch, epochs):
    """
    Applies cosine decay to the learning rate.

    when training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, the learning rate will be decayed by
    following cosine decay strategy.

    .. math::

        decayed\_lr = learning\_rate * 0.5 * (math.cos * (epoch * \\frac{math.pi}{epochs} ) + 1)

    Args:
        learning_rate(Variable|float): The initial learning rate.
        step_each_epoch(int): the number of steps in an epoch.
        epochs(int): the number of epochs.

    Returns:
        Variable: The decayed learning rate.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            base_lr = 0.1
            lr = fluid.layers.cosine_decay(
            learning_rate = base_lr, step_each_epoch=10000, epochs=120)
    """
    check_type(learning_rate, 'learning_rate', (float, tensor.Variable),
               'cosine_decay')

    with default_main_program()._lr_schedule_guard():
        if in_dygraph_mode():
            decay = imperate_lr.CosineDecay(learning_rate, step_each_epoch,
                                            epochs)
            return decay
        else:
            global_step = _decay_step_counter()

            cur_epoch = ops.floor(global_step / step_each_epoch)
            decayed_lr = learning_rate * 0.5 * (
                ops.cos(cur_epoch * math.pi / epochs) + 1)
            return decayed_lr


def linear_lr_warmup(learning_rate, warmup_steps, start_lr, end_lr):
    """
    This operator use the linear learning rate warm up strategy to adjust the learning rate preliminarily before the normal learning rate scheduling.
    For more information, please refer to `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_
    
    When global_step < warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            linear_step = end_lr - start_lr
            lr = start_lr + linear_step * (global_step / warmup_steps)
    
    where start_lr is the initial learning rate, and end_lr is the final learning rate;
    
    When global_step >= warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            lr = learning_rate
    
    where lr is the learning_rate after warm-up.
    
    Args:
        learning_rate (Variable|float): Learning_rate after warm-up, it could be 1D-Tensor or single value with the data type of float32.
        warmup_steps (int): Steps for warm up.
        start_lr (float): Initial learning rate of warm up.
        end_lr (float): Final learning rate of warm up.
    
    Returns:
        Variable: Warm-up learning rate with the same data type as learning_rate.
    
    
    Examples:
    
    .. code-block:: python
    
        import paddle.fluid as fluid
    
        boundaries = [100, 200]
        lr_steps = [0.1, 0.01, 0.001]
        learning_rate = fluid.layers.piecewise_decay(boundaries, lr_steps) #case1, 1D-Tensor
        #learning_rate = 0.1  #case2, single-value
        warmup_steps = 50
        start_lr = 1. / 3.
        end_lr = 0.1
        decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
            warmup_steps, start_lr, end_lr)
    
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        out, = exe.run(fetch_list=[decayed_lr.name])
        print(out)
        # case1: [0.33333334]
        # case2: [0.33333334]
    """
    dtype = 'float32'
    if isinstance(learning_rate, Variable):
        dtype = learning_rate.dtype

    linear_step = float(end_lr) - float(start_lr)
    with default_main_program()._lr_schedule_guard():

        if in_dygraph_mode():
            lr = imperate_lr.LinearLrWarmup(learning_rate, warmup_steps,
                                            start_lr, end_lr)
            return lr
        else:
            lr = tensor.create_global_var(
                shape=[1],
                value=0.0,
                dtype=dtype,
                persistable=True,
                name="learning_rate_warmup")

            global_step = _decay_step_counter()

            with control_flow.Switch() as switch:
                with switch.case(global_step < warmup_steps):
                    decayed_lr = start_lr + linear_step * (global_step /
                                                           float(warmup_steps))
                    tensor.assign(decayed_lr, lr)
                with switch.default():
                    if not isinstance(learning_rate, Variable):
                        learning_rate = tensor.fill_constant(
                            shape=[1], dtype=dtype, value=float(learning_rate))
                    tensor.assign(learning_rate, lr)
            return lr


def reduce_lr_on_plateau(indicator,
                         learning_rate,
                         mode='min',
                         decay_rate=0.1,
                         patience=10,
                         threshold=1e-4,
                         threshold_mode='rel',
                         cooldown=0,
                         min_lr=0):
    """
    Reduce learning rate when the monitoring ``indicator`` has stopped improving. Models often benefit 
    from reducing the learning rate by 2 to 10 times once learning stagnates.

    ``indicator`` will be monitored to determine whether the learning rate will reduce. In ``'min'`` mode, 
    when ``indicator`` stop decreasing for a ``patience`` number of epochs, the learning rate will be 
    reduced to ``learning_rate * decay_rate`` . In ``'max'`` mode, all is the same when ``indicator`` stop increasing. 
    
    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming normal operation.

    Args:
        indicator (Variable): A ``Variable`` that will be monitored to determine whether the learning rate will reduce. 
            If it has no improvement, the learning rate will reduce. It should be 1-D Tensor with shape [1]. 
            Usually, it is the ``loss`` in the network.
        learning_rate (Variable|float|int): The initial learning rate. It can be set to python float or int number.
            If the type is Variable, it should be 1-D Tensor with shape [1], the data type can be 'float32' or 'float64'.
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. ``'min'`` means that the learning rate will reduce when 
            ``indicator`` stops decreasing, and ``'max'`` means that the learning rate will reduce when ``indicator`` stops 
            increasing. Default: ``'min'`` .
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` . 
            It should be less than 1.0. Default: 0.1.
        patience (int, optional): When ``indicator`` doesn't improve for this number of epochs, learing rate will be reduced. 
            Default: 10.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``indicator`` . 
            This make tiny changes of ``indicator`` will be ignored. Default: 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``indicator``
            is ``last_loss * threshold`` , where ``last_loss`` is the value of ``indicator`` in last epoch. In ``'abs'`` mode, 
            the minimum change of ``indicator`` is ``threshold`` . Default: ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Default: 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Default: 0.
    
    Returns:
        Variable: The learning rate determined by the improvement of 'indicator'

    Examples:
    
    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        x = fluid.data(name='X', shape=[10, 10])
        predict = fluid.layers.fc(x, 10)
        loss = fluid.layers.reduce_mean(predict)

        lr = fluid.layers.reduce_lr_on_plateau(
                                indicator = loss, 
                                learning_rate = 1.0,
                                decay_rate = 0.5,
                                patience = 5,
                                cooldown = 2)

        adam = fluid.optimizer.Adam(learning_rate = lr)
        adam.minimize(loss)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        for epoch in range(20):
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            out = exe.run(fluid.default_main_program(), feed={'X':x}, fetch_list=[loss, lr])
            print("current loss is %s, current lr is %s" % (out[0], out[1]))        
    """
    dtype = 'float32'
    if isinstance(learning_rate, Variable):
        dtype = learning_rate.dtype

    with default_main_program()._lr_schedule_guard():

        if in_dygraph_mode():
            lr = imperate_lr.ReduceLROnPlateau(learning_rate, mode, decay_rate,
                                               patience, threshold,
                                               threshold_mode, cooldown, min_lr)
            return lr
        else:
            check_type(indicator, 'indicator', Variable, 'reduce_lr_on_plateau')
            check_type(learning_rate, 'learning_rate', (float, int, Variable),
                       'reduce_lr_on_plateau')
            assert len(indicator.shape) == 1 and indicator.shape[0] == 1, "The shape of monitor "   \
                "indicator in reduce_lr_on_plateau should be (1L,), but the current shape is {}. "  \
                "Maybe that you should call fluid.layers.mean to process the indicator in advance." \
                .format(indicator.shape)

            assert indicator.shape == (
                1,
            ), "The monitor indicator in reduce_lr_on_plateau must be 1-D tensor with shape [1]"

            if decay_rate >= 1.0:
                raise ValueError(
                    'new_lr = origin_lr * decay_rate and decay_rate should be < 1.0.'
                )

            if mode not in ['min', 'max']:
                raise ValueError('mode ' + mode + ' is unknown!')

            if threshold_mode not in ['rel', 'abs']:
                raise ValueError('threshold mode ' + threshold_mode +
                                 ' is unknown!')

            global_step = _decay_step_counter()
            best = tensor.create_global_var(
                name="best",
                shape=[1],
                value=0,
                dtype=indicator.dtype,
                persistable=True)

            def init_best_loss():
                init_var = (indicator + 1) if mode == 'min' else (indicator - 1)
                tensor.assign(init_var, best)

            control_flow.cond(global_step == 0, init_best_loss)

            if mode == 'min' and threshold_mode == 'rel':
                better_cond = control_flow.less_than(indicator,
                                                     best - best * threshold)

            elif mode == 'min' and threshold_mode == 'abs':
                better_cond = control_flow.less_than(indicator,
                                                     best - threshold)

            elif mode == 'max' and threshold_mode == 'rel':
                better_cond = control_flow.greater_than(indicator,
                                                        best + best * threshold)

            else:  # mode == 'max' and epsilon_mode == 'abs':
                better_cond = control_flow.greater_than(indicator,
                                                        best - threshold)

            if not isinstance(learning_rate, tensor.Variable):
                learning_rate = tensor.create_global_var(
                    name="learning_rate_reduce",
                    shape=[1],
                    value=float(learning_rate),
                    dtype=dtype,
                    persistable=True)

            cooldown = tensor.fill_constant(
                shape=[1], dtype='int32', value=cooldown)
            min_lr = tensor.fill_constant(
                shape=[1], dtype='float32', value=min_lr)
            zero = tensor.zeros(shape=[1], dtype='int32')

            cooldown_counter = tensor.create_global_var(
                name="cooldown_counter",
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

            num_bad_epochs = tensor.create_global_var(
                name="num_bad_epochs",
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

            def true_func():
                tensor.assign(zero, num_bad_epochs)
                tensor.assign(cooldown, cooldown_counter)
                new_lr = nn.elementwise_max(learning_rate * decay_rate, min_lr)
                tensor.assign(new_lr, learning_rate)

            # the first level control_flow
            with control_flow.Switch() as cooldown_switch:
                with cooldown_switch.case(cooldown_counter > 0):
                    control_flow.increment(cooldown_counter, value=-1)
                with cooldown_switch.default():
                    # the second level control_flow
                    with control_flow.Switch() as better_switch:
                        with better_switch.case(better_cond):
                            tensor.assign(indicator, best)
                            tensor.assign(zero, num_bad_epochs)
                        with better_switch.default():
                            control_flow.increment(num_bad_epochs)

                            # the thirdth level control_flow
                            control_flow.cond(num_bad_epochs > patience,
                                              true_func)

            return learning_rate
