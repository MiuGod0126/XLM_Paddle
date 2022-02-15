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

import copy
import io
import json
import os
import six
import logging
import inspect
import paddle
from paddle.nn import Layer
# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddlenlp.utils.downloader import get_path_from_url, COMMUNITY_MODEL_PREFIX
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger

from .generation_utils import GenerationMixin
from .utils import InitTrackerMeta, fn_args_to_dict

__all__ = [
    'PretrainedModel',
    'register_base_model',
    'apply_chunking_to_forward',
    'find_pruneable_heads_and_indices',
    'prune_linear_layer',
]



def register_base_model(cls):
    """
    A decorator for `PretrainedModel` class. It first retrieves the parent class
    of the class being decorated, then sets the `base_model_class` attribute
    of that parent class to be the class being decorated. In summary, the decorator registers
    the decorated class as the base model class in all derived classes under the same architecture.

    Args:
        cls (PretrainedModel): The class (inherited from PretrainedModel) to be decorated .

    Returns:
        PretrainedModel: The input class `cls` after decorating.

    Example:
        .. code-block::

            from paddlenlp.transformers import BertModel, register_base_model

            BertModel = register_base_model(BertModel)
            assert BertModel.base_model_class == BertModel
    """
    base_cls = cls.__bases__[0]
    assert issubclass(
        base_cls, PretrainedModel
    ), "`register_base_model` should be used on subclasses of PretrainedModel."
    base_cls.base_model_class = cls
    return cls


@six.add_metaclass(InitTrackerMeta)
class PretrainedModel(Layer, GenerationMixin):
    """
    The base class for all pretrained models. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **model_config_file** (str): Represents the file name of model configuration
      for configuration saving and loading in local file system. The value is
      `model_config.json`.
    - **resource_files_names** (dict): Name of local file where the model configuration
      can be saved and loaded locally. Currently, resources only include the model state,
      thus the dict only includes `'model_state'` as key with corresponding
      value `'model_state.pdparams'` for model weights saving and loading.
    - **pretrained_init_configuration** (dict): Provides the model configurations
      of built-in pretrained models (contrasts to models in local file system).
      It has pretrained model names as keys (such as `bert-base-uncased`), and
      the values are dict preserving corresponding configuration for model initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained models (contrasts to models in local file system).
      It has the same key as resource_files_names (that is "model_state"),
      and the corresponding value is a dict with specific model name to model weights URL mapping
      (such as "bert-base-uncased" ->
      "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams").
    - **base_model_prefix** (str): Represents the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model. Note: A base model class is pretrained model class
      decorated by `register_base_model`, such as `BertModel`; A derived model
      class is a pretrained model class adding layers on top of the base model,
      and it has a base model as attribute, such as `BertForSequenceClassification`.

    Methods common to models for text generation are defined in `GenerationMixin`
    and also inherited here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedModel`,
    by which subclasses can track arguments for initialization automatically.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fields as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {}
    base_model_prefix = ""

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the pretrained model instance.
        """
        init_dict = fn_args_to_dict(original_init, *((self, ) + args), **kwargs)
        self.config = init_dict

    @property
    def base_model(self):
        """
        PretrainedModel: The body of the same model architecture. It is the base
            model itself for base model or the base model attribute for derived
            model.
        """
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        """
        list: Contains all supported built-in pretrained model names of the
            current PretrainedModel class.
        """
        # Todo: return all model name
        return list(self.pretrained_init_configuration.keys())

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None  # Overwrite for models with output embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/')
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}

        # From built-in pretrained models
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[
                    pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(pretrained_model_name_or_path,
                                              file_name)
                resource_files[file_id] = full_file_name
            resource_files["model_config_file"] = os.path.join(
                pretrained_model_name_or_path, cls.model_config_file)
        else:
            # Assuming from community-contributed pretrained models
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(COMMUNITY_MODEL_PREFIX,
                                              pretrained_model_name_or_path,
                                              file_name)
                resource_files[file_id] = full_file_name
            resource_files["model_config_file"] = os.path.join(
                COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path,
                cls.model_config_file)

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
                continue
            path = os.path.join(default_root, file_path.split('/')[-1])
            if os.path.exists(path):
                logger.info("Already cached %s" % path)
                resolved_resource_files[file_id] = path
            else:
                logger.info("Downloading %s and saved to %s" %
                            (file_path, default_root))
                try:
                    resolved_resource_files[file_id] = get_path_from_url(
                        file_path, default_root)
                except RuntimeError as err:
                    logger.error(err)
                    raise RuntimeError(
                        f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                        f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                        "- a correct model-identifier of built-in pretrained models,\n"
                        "- or a correct model-identifier of community-contributed pretrained models,\n"
                        "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                    )

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file",
                                                        None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class",
                                     cls.base_model_class.__name__)
        # Check if the loaded config matches the current model class's __init__
        # arguments. If not match, the loaded config is for the base model class.
        if init_class == cls.base_model_class.__name__:
            base_args = init_args
            base_kwargs = init_kwargs
            derived_args = ()
            derived_kwargs = {}
            base_arg_index = None
        else:  # extract config for base model
            derived_args = list(init_args)
            derived_kwargs = init_kwargs
            base_arg = None
            for i, arg in enumerate(init_args):
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = i
                    base_arg = arg
                    break
            for arg_name, arg in init_kwargs.items():
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = arg_name
                    base_arg = arg
                    break

            base_args = base_arg.pop("init_args", ())
            base_kwargs = base_arg
        if cls == cls.base_model_class:
            # Update with newly provided args and kwargs for base model
            base_args = base_args if not args else args
            base_kwargs.update(kwargs)
            model = cls(*base_args, **base_kwargs)
        else:
            # Update with newly provided args and kwargs for derived model
            base_parameters_dict = inspect.signature(
                cls.base_model_class.__init__).parameters
            for k, v in kwargs.items():
                if k in base_parameters_dict:
                    base_kwargs[k] = v
            base_model = cls.base_model_class(*base_args, **base_kwargs)
            if base_arg_index is not None:
                derived_args[base_arg_index] = base_model
            else:
                derived_args = (base_model, )  # assume at the first position
            derived_args = derived_args if not args else args
            derived_parameters_dict = inspect.signature(cls.__init__).parameters
            for k, v in kwargs.items():
                if k in derived_parameters_dict:
                    derived_kwargs[k] = v
            model = cls(*derived_args, **derived_kwargs)

        # Maybe need more ways to load resources.
        weight_path = resolved_resource_files["model_state"]
        assert weight_path.endswith(
            ".pdparams"), "suffix of weight must be .pdparams"

        state_dict = paddle.load(weight_path)

        # Make sure we are able to load base models as well as derived models
        # (with heads)
        start_prefix = ""
        model_to_load = model
        state_to_load = state_dict
        unexpected_keys = []
        missing_keys = []
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # base model
            state_to_load = {}
            start_prefix = cls.base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(cls.base_model_prefix):
                    state_to_load[k[len(start_prefix):]] = v
                else:
                    unexpected_keys.append(k)
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # derived model (base model with heads)
            model_to_load = getattr(model, cls.base_model_prefix)
            for k in model.state_dict().keys():
                if not k.startswith(cls.base_model_prefix):
                    missing_keys.append(k)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".
                        format(model.__class__.__name__, unexpected_keys))
        if paddle.in_dynamic_mode():
            model_to_load.set_state_dict(state_to_load)
            return model
        return model, state_to_load

    def save_model_config(self, save_dir):
        """
        Saves model configuration to a file named "model_config.json" under `save_dir`.

        Args:
            save_dir (str): Directory to save model_config file into.
        """
        # Save model config
        model_config_file = os.path.join(save_dir, self.model_config_file)
        model_config = self.init_config
        # If init_config contains a Layer, use the layer's init_config to save
        for key, value in model_config.items():
            if key == "init_args":
                args = []
                for arg in value:
                    args.append(
                        arg.init_config
                        if isinstance(arg, PretrainedModel) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, PretrainedModel):
                model_config[key] = value.init_config
        with io.open(model_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_config, ensure_ascii=False))

    def save_pretrained(self, save_dir):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """
        assert not os.path.isfile(
            save_dir
        ), "Saving directory ({}) should be a directory, not a file".format(
            save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Save model config
        self.save_model_config(save_dir)
        # Save model
        if paddle.in_dynamic_mode():
            file_name = os.path.join(
                save_dir, list(self.resource_files_names.values())[0])
            paddle.save(self.state_dict(), file_name)
        else:
            logger.warning(
                "Save pretrained model only supported dygraph mode for now!")



def apply_chunking_to_forward(
    forward_fn, chunk_size, chunk_dim, *input_tensors):
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., paddle.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[paddle.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`paddle.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, axis=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return paddle.concat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


def find_pruneable_heads_and_indices(
    heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], paddle.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = paddle.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.reshape((-1)).equal(1)
    index= paddle.cast(paddle.arange(len(mask))[mask],dtype='int64')
    return heads, index


def prune_linear_layer(layer, index, axis):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`paddle.nn.Linear`): The layer to prune.
        index (:obj:`paddle.LongTensor`): The indices to keep in the layer.
        axis (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`paddle.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    # index = index.to(layer.weight.device)
    W = layer.weight.index_select(index=index,axis=axis).clone().detach()
    if layer.bias is not None:
        if axis == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.shape)
    new_size[axis] = len(index)
    new_layer = paddle.nn.Linear(new_size[1], new_size[0], bias_attr=layer.bias is not None)
    new_layer.weight.stop_gradient = True
    new_layer.weight.set_value(W)
    new_layer.weight.stop_gradient = False
    if layer.bias is not None:
        new_layer.bias.stop_gradient = True
        new_layer.bias.set_value(b)
        new_layer.bias.stop_gradient = False
    return new_layer

