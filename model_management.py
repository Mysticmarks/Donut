import os
import importlib # For dynamic class loading
import json # For parsing model_spec.json

import torch
from accelerate import init_empty_weights
from accelerate.utils import infer_auto_device_map, load_checkpoint_and_dispatch

from modeling.autoencoder import load_ae
from modeling.bagel import BagelConfig, Bagel 
from utils.image_transform import ImageTransform 
from utils.other import add_special_tokens
from modeling.abs_models import BaseLLMWrapper, BaseVisionWrapper # Abstract base classes

# --- Constants ---
MODEL_REGISTRY_PATH = "available_models"

# --- Model Wrappers ---
# These wrappers adapt specific model architectures (like Qwen2, Siglip)
# to the generic interfaces (BaseLLMWrapper, BaseVisionWrapper) expected by the Bagel model.

class Qwen2Wrapper(BaseLLMWrapper):
    """
    Wrapper for Qwen2-based language models.
    """
    def __init__(self, model, config):
        super().__init__(model, config)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

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
        """
        Calls the underlying Qwen2ForCausalLM's forward_inference method.
        The `attention_mask` (if provided in `extra_inputs`) is passed through.
        """
        return self.model.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            **extra_inputs  # Pass through any other arguments, including attention_mask
        )

    def get_lm_head(self) -> torch.nn.Module:
        return self.model.lm_head


class SiglipWrapper(BaseVisionWrapper):
    """
    Wrapper for Siglip-based vision models.
    """
    def __init__(self, model, config):
        super().__init__(model, config)

    def process_images(
        self, 
        packed_pixel_values: torch.Tensor, 
        packed_flattened_position_ids: torch.Tensor, 
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs # For arguments like output_hidden_states
    ) -> torch.Tensor:
        """
        Processes images using the underlying SiglipVisionModel.
        """
        return self.model(
            pixel_values=packed_pixel_values,
            flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs
        )

# --- Model Discovery ---
def discover_available_models() -> list[dict]:
    """
    Scans the MODEL_REGISTRY_PATH for model_spec.json files.
    Returns a list of dictionaries, each containing 'id' and 'name' for a discovered model.
    """
    available_models = []
    if not os.path.exists(MODEL_REGISTRY_PATH):
        print(f"Warning: Model registry path '{MODEL_REGISTRY_PATH}' not found.")
        return available_models

    for model_id in os.listdir(MODEL_REGISTRY_PATH):
        model_dir = os.path.join(MODEL_REGISTRY_PATH, model_id)
        if os.path.isdir(model_dir):
            spec_path = os.path.join(model_dir, "model_spec.json")
            if os.path.exists(spec_path):
                try:
                    with open(spec_path, 'r') as f:
                        spec = json.load(f)
                    available_models.append({
                        "id": model_id,
                        "name": spec.get("display_name", model_id) # Use display_name or model_id as fallback
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse 'model_spec.json' for model '{model_id}' at {spec_path}.")
            else:
                print(f"Warning: 'model_spec.json' not found in model directory: {model_dir}")
    return available_models

# --- Dynamic Model Loading ---
def _import_class(class_reference_str: str):
    """
    Helper function to dynamically import a class given its full reference string.
    Example: "modeling.qwen2.Qwen2ForCausalLM"
    """
    try:
        module_path, class_name = class_reference_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class {class_reference_str}: {e}")


def load_model_and_tokenizer(model_id: str = "BAGEL-7B-MoT") -> tuple:
    """
    Loads a specified model and its components (tokenizer, transforms) 
    based on its 'model_spec.json' configuration.

    Args:
        model_id: The identifier of the model to load, corresponding to a subdirectory
                  in MODEL_REGISTRY_PATH.

    Returns:
        A tuple containing: (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    
    Raises:
        FileNotFoundError: If the model_spec.json for the given model_id is not found.
        ImportError: If specified classes cannot be imported.
    """
    spec_path = os.path.join(MODEL_REGISTRY_PATH, model_id, "model_spec.json")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"'model_spec.json' for model_id '{model_id}' not found at {spec_path}.")

    with open(spec_path, 'r') as f:
        spec = json.load(f)

    # Extract information from the specification file
    paths = spec["paths"]
    class_refs = spec["class_references"]
    add_configs = spec.get("additional_configs", {})
    model_weights_root = paths["model_weights_root"] # Base path for model weights and configs

    # Dynamically import necessary classes
    LLMConfigClass = _import_class(class_refs["llm_config_class"])
    LLMClass = _import_class(class_refs["llm_class"])
    VisionConfigClass = _import_class(class_refs["vision_config_class"])
    VisionClass = _import_class(class_refs["vision_class"])
    LLMWrapperClass = _import_class(class_refs["llm_wrapper_class"])
    VisionWrapperClass = _import_class(class_refs["vision_wrapper_class"])
    TokenizerClass = _import_class(class_refs["tokenizer_class"])

    # 1. Load LLM Configuration
    llm_config_file = os.path.join(model_weights_root, paths["llm_config"])
    if os.path.exists(llm_config_file) and paths["llm_config"].endswith(".json"):
        llm_config = LLMConfigClass.from_json_file(llm_config_file)
    else: # Fallback to directory-based loading (e.g. from_pretrained with subfolder)
        llm_config_source = os.path.join(model_weights_root, os.path.dirname(paths["llm_config"])) if '/' in paths["llm_config"] else model_weights_root
        subfolder_name = os.path.basename(paths["llm_config"]) if not paths["llm_config"].endswith(".json") else None
        if subfolder_name == paths["llm_config"]: # if paths["llm_config"] is just a subfolder name
             llm_config_source = model_weights_root
             subfolder_name = paths["llm_config"]

        llm_config = LLMConfigClass.from_pretrained(llm_config_source, subfolder=subfolder_name if subfolder_name else None)

    # 2. Load Vision Model Configuration
    vit_config_file = os.path.join(model_weights_root, paths["vit_config"])
    if os.path.exists(vit_config_file) and paths["vit_config"].endswith(".json"):
        vit_config = VisionConfigClass.from_json_file(vit_config_file)
    else:
        vit_config_source = os.path.join(model_weights_root, os.path.dirname(paths["vit_config"])) if '/' in paths["vit_config"] else model_weights_root
        subfolder_name = os.path.basename(paths["vit_config"]) if not paths["vit_config"].endswith(".json") else None
        if subfolder_name == paths["vit_config"]:
             vit_config_source = model_weights_root
             subfolder_name = paths["vit_config"]
        vit_config = VisionConfigClass.from_pretrained(vit_config_source, subfolder=subfolder_name if subfolder_name else None)

    # 3. Apply additional configurations specified in the model_spec
    for key, value in add_configs.get("llm_config_override", {}).items(): setattr(llm_config, key, value)
    for key, value in add_configs.get("vit_config_override", {}).items(): setattr(vit_config, key, value)
    if "vit_num_hidden_layers_reduction" in add_configs: 
        vit_config.num_hidden_layers -= add_configs["vit_num_hidden_layers_reduction"]
    
    # 4. Load VAE (Autoencoder)
    vae_subfolder = os.path.dirname(paths.get("vae_weights", "vae")) # Default to 'vae' if not specified
    vae_local_path = os.path.join(model_weights_root, paths.get("vae_weights", "vae/ae.safetensors"))
    vae_model = load_ae(model_weights_root, subfolder=vae_subfolder, torch_dtype=torch.bfloat16, local_path=vae_local_path)
    vae_config = vae_model.config

    # 5. Create Bagel Model Configuration
    bagel_config_params = {
        "llm_config": llm_config, "vit_config": vit_config, "vae_config": vae_config,
        "visual_gen": add_configs.get("bagel_visual_gen", True),
        "visual_und": add_configs.get("bagel_visual_und", True),
        "vit_max_num_patch_per_side": add_configs.get("bagel_vit_max_num_patch_per_side", 70),
        "connector_act": add_configs.get("bagel_connector_act", 'gelu_pytorch_tanh'),
        "latent_patch_size": add_configs.get("bagel_latent_patch_size", 2),
        "max_latent_size": add_configs.get("bagel_max_latent_size", 64),
    }
    bagel_config = BagelConfig(**bagel_config_params)
    
    # 6. Instantiate actual LLM and Vision models, then wrap them
    llm_model_instance = LLMClass(llm_config)
    vit_model_instance = VisionClass(vit_config)

    # Apply specific model modifications if any (e.g., for ViT embeddings)
    if add_configs.get("vit_convert_conv2d_to_linear", False):
        if hasattr(vit_model_instance, 'vision_model') and \
           hasattr(vit_model_instance.vision_model, 'embeddings') and \
           hasattr(vit_model_instance.vision_model.embeddings, 'convert_conv2d_to_linear'):
            vit_model_instance.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    llm_wrapper = LLMWrapperClass(model=llm_model_instance, config=llm_config)
    vit_wrapper = VisionWrapperClass(model=vit_model_instance, config=vit_config)
    
    # 7. Instantiate the main Bagel model
    with init_empty_weights(): # Initialize Bagel with empty weights before loading checkpoint
        model = Bagel(language_model=llm_wrapper, vision_tower=vit_wrapper, config=bagel_config)

    # 8. Load Tokenizer
    tokenizer_source_path = paths["tokenizer_path"]
    if tokenizer_source_path == ".": # Indicates tokenizer is in the root of model_weights_root
        tokenizer_load_path = model_weights_root
    else: # Assumes it's a subfolder or specific file path relative to model_weights_root
        tokenizer_load_path = os.path.join(model_weights_root, tokenizer_source_path)
        if not os.path.isdir(tokenizer_load_path): # If not a directory, assume it's a subfolder containing tokenizer files
            tokenizer_load_path = os.path.dirname(tokenizer_load_path)
            
    tokenizer = TokenizerClass.from_pretrained(tokenizer_load_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer, model=model) 
    
    # 9. Initialize Image Transforms
    # These could also be part of the model_spec if they vary significantly per model.
    vae_transform = ImageTransform(
        vae_config.image_size, mean=vae_config.image_mean, std=vae_config.image_std, is_vae=True
    )
    vit_transform = ImageTransform(
        vit_config.image_size, mean=vit_config.image_mean, std=vit_config.image_std
    )

    # 10. Prepare for Multi-GPU Inference and Load Checkpoint
    no_split_classes = ["Bagel", LLMWrapperClass.__name__, VisionWrapperClass.__name__]
    if "llm_decoder_layer_class_name" in add_configs: # e.g. "Qwen2MoTDecoderLayer"
        no_split_classes.append(add_configs["llm_decoder_layer_class_name"])
    
    # Max memory configuration should ideally be more dynamic or part of the spec
    max_memory_config = add_configs.get("max_memory", {0: "20GiB", 1: "20GiB", "cpu": "32GiB"})

    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=no_split_classes,
        max_memory=max_memory_config
    )

    checkpoint_file_path = os.path.join(model_weights_root, paths["checkpoint_file"])
    load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_file_path,
        device_map=device_map,
        offload_folder=add_configs.get("offload_folder", "offload"), 
        offload_state_dict=add_configs.get("offload_state_dict", True),
        dtype=torch.bfloat16, # This could be part of the spec
        force_hooks=add_configs.get("force_hooks", True)
    )
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
