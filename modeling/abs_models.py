import abc
import torch


class BaseLLMWrapper(abc.ABC):
    def __init__(self, model, config): # Order: model, then config as per typical Hugging Face style
        self.model = model
        self._config = config 

    @abc.abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward_inference(
        self, 
        packed_query_sequence: torch.Tensor, 
        query_lens: list[int], 
        packed_query_position_ids: torch.Tensor, 
        packed_query_indexes: torch.Tensor, 
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]], 
        packed_key_value_indexes: torch.Tensor, 
        key_values_lens: list[int], 
        update_past_key_values: bool, 
        is_causal: bool, 
        **extra_inputs
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor]]]:
        pass

    @abc.abstractmethod
    def get_lm_head(self) -> torch.nn.Module:
        pass

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return self.model.device


class BaseVisionWrapper(abc.ABC):
    def __init__(self, model, config): # Order: model, then config
        self.model = model
        self._config = config

    @abc.abstractmethod
    def process_images(
        self, 
        packed_pixel_values: torch.Tensor, 
        packed_flattened_position_ids: torch.Tensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int,
        **kwargs 
    ) -> torch.Tensor:
        pass

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return self.model.device
