import gradio as gr
import numpy as np
import os
import torch
import random

from PIL import Image

from data.data_utils import pil_img2rgb # Removed add_special_tokens
# from data.transforms import ImageTransform # This is now in model_management.py
from inferencer import InterleaveInferencer
# Imports moved to model_management.py:
# from modeling.autoencoder import load_ae
# from modeling.bagel.qwen2_navit import NaiveCache
# from modeling.bagel import (
#     BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
#     SiglipVisionConfig, SiglipVisionModel
# )
# from modeling.qwen2 import Qwen2Tokenizer
# from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Import model discovery and loading functions
from model_management import discover_available_models, load_model_and_tokenizer

# --- Phase 2: Initial Model Load and Discovery ---
# Discover available models for the UI
available_models_list = discover_available_models()
DEFAULT_MODEL_ID = "BAGEL-7B-MoT" # Default model

# Global variables for model components and inferencer
model = None
vae_model = None
tokenizer = None
vae_transform = None
vit_transform = None
new_token_ids = None
inferencer = None
current_model_display_name = ""

def reload_model_and_inferencer(model_id_to_load):
    global model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, inferencer, current_model_display_name
    
    print(f"Attempting to load model: {model_id_to_load}")
    try:
        new_model_tuple = load_model_and_tokenizer(model_id=model_id_to_load)
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = new_model_tuple
        
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        # Update current model display name for UI
        selected_model_info = next((m for m in available_models_list if m["id"] == model_id_to_load), None)
        if selected_model_info:
            current_model_display_name = selected_model_info["name"]
        else:
            current_model_display_name = model_id_to_load # Fallback if not in list
        print(f"Successfully loaded and re-initialized for model: {model_id_to_load}")
        return f"Successfully loaded model: {current_model_display_name}"
    except Exception as e:
        print(f"Error loading model {model_id_to_load}: {e}")
        # Attempt to reload the previous model if possible, or handle error state
        # For now, just print error and return status
        return f"Error loading model {model_id_to_load}: {e}"

# Load the default model on startup
initial_load_status = reload_model_and_inferencer(DEFAULT_MODEL_ID)
print(initial_load_status)
# --- End of Phase 2: Initial Model Load ---


# Inferencer Preparing (Now handled within reload_model_and_inferencer)
# inferencer = InterleaveInferencer(
#     model=model,
#     vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

# Text to Image function with thinking option and hyperparameters
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, 
                 timestep_shift=3.0, num_timesteps=50, 
                 cfg_renorm_min=1.0, cfg_renorm_type="global", 
                 max_think_token_n=1024, do_sample=False, text_temperature=0.3,
                 seed=0, image_ratio="1:1"):
    # Set seed for reproducibility
    set_seed(seed)

    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768) 
    elif image_ratio == "16:9":
        image_shapes = (576, 1024)
    elif image_ratio == "9:16":
        image_shapes = (1024, 576) 
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    
    # Call inferencer with or without think parameter based on user choice
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", None)


# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False, 
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )
    
    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking, 
                        understanding_output=True, **inference_hyper)
    return result["text"]


# Image Editing function with thinking option and hyperparameters
def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, 
              cfg_img_scale=2.0, cfg_interval=0.0, 
              timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=1.0, 
              cfg_renorm_type="text_channel", max_think_token_n=1024, 
              do_sample=False, text_temperature=0.3, seed=0):
    # Set seed for reproducibility
    set_seed(seed)
    
    if image is None:
        return "Please upload an image.", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Include thinking parameter based on user choice
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", "")


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


# --- Gradio UI Functions ---

def create_text_to_image_tab():
    with gr.Tab("📝 Text to Image"):
        with gr.Row():
            with gr.Column(scale=2):
                txt_input_t2i = gr.Textbox(
                    label="Prompt", 
                    value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress "
                          "made of delicate fabrics in soft, mystical colors like emerald green and silver. "
                          "She has pointed ears, a gentle, enchanting expression, and her outfit is adorned "
                          "with sparkling jewels and intricate patterns. The background is a magical forest "
                          "with glowing plants, mystical creatures, and a serene atmosphere."
                )
                thinking_output_t2i = gr.Textbox(label="Thinking Process", visible=False, lines=3)
            with gr.Column(scale=1):
                img_output_t2i = gr.Image(label="Generated Image")

        with gr.Row():
            show_thinking_t2i = gr.Checkbox(label="Thinking", value=False)
        
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    seed_t2i = gr.Slider(
                        minimum=0, maximum=1000000, value=0, step=1, 
                        label="Seed", info="0 for random, >0 for reproducible"
                    )
                    image_ratio_t2i = gr.Dropdown(
                        choices=["1:1", "4:3", "3:4", "16:9", "9:16"], value="1:1", 
                        label="Image Ratio", info="Longer side fixed to 1024"
                    )
                with gr.Row():
                    cfg_text_scale_t2i = gr.Slider(1.0, 8.0, 4.0, step=0.1, label="CFG Text Scale")
                    cfg_interval_t2i = gr.Slider(0.0, 1.0, 0.4, step=0.1, label="CFG Interval Start")
                with gr.Row():
                    cfg_renorm_type_t2i = gr.Dropdown(["global", "local", "text_channel"], value="global", label="CFG Renorm Type")
                    cfg_renorm_min_t2i = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Renorm Min")
                with gr.Row():
                    num_timesteps_t2i = gr.Slider(10, 100, 50, step=5, label="Timesteps")
                    timestep_shift_t2i = gr.Slider(1.0, 5.0, 3.0, step=0.5, label="Timestep Shift")
                
                thinking_params_t2i = gr.Group(visible=False)
                with thinking_params_t2i:
                    with gr.Row():
                        do_sample_t2i = gr.Checkbox(label="Sampling (Text)", value=False)
                        max_think_token_n_t2i = gr.Slider(64, 4006, 1024, step=64, label="Max Think Tokens")
                        text_temperature_t2i = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Temperature (Text)")
        
        gen_btn_t2i = gr.Button("Generate Image")
        
        def ui_update_thinking_visibility(show: bool):
            return gr.update(visible=show), gr.update(visible=show)
        
        show_thinking_t2i.change(
            fn=ui_update_thinking_visibility, inputs=[show_thinking_t2i],
            outputs=[thinking_output_t2i, thinking_params_t2i]
        )
        
        def process_t2i_request(
            prompt, show_thinking, cfg_text_scale, cfg_interval, timestep_shift, 
            num_timesteps, cfg_renorm_min, cfg_renorm_type, 
            max_think_token_n, do_sample, text_temperature, seed_val, image_ratio_val
        ):
            img, thinking_text = text_to_image( # Calls the global text_to_image
                prompt, show_thinking, cfg_text_scale, cfg_interval,
                timestep_shift, num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed_val, image_ratio_val
            )
            return img, thinking_text if thinking_text else ""
        
        gen_btn_t2i.click(
            fn=process_t2i_request,
            inputs=[
                txt_input_t2i, show_thinking_t2i, cfg_text_scale_t2i, 
                cfg_interval_t2i, timestep_shift_t2i, num_timesteps_t2i, 
                cfg_renorm_min_t2i, cfg_renorm_type_t2i, max_think_token_n_t2i, 
                do_sample_t2i, text_temperature_t2i, seed_t2i, image_ratio_t2i
            ],
            outputs=[img_output_t2i, thinking_output_t2i]
        )
    return txt_input_t2i # Return a representative component if needed for other interactions, or None


def create_image_edit_tab():
    with gr.Tab("🖌️ Image Edit"):
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_input_ie = gr.Image(
                    label="Input Image", value=load_example_image('test_images/women.jpg')
                )
                edit_prompt_ie = gr.Textbox(
                    label="Edit Prompt",
                    value="She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."
                )
            with gr.Column(scale=1):
                edit_image_output_ie = gr.Image(label="Edited Image")
                edit_thinking_output_ie = gr.Textbox(label="Thinking Process", visible=False, lines=3)
        
        with gr.Row():
            edit_show_thinking_ie = gr.Checkbox(label="Thinking", value=False)
        
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    edit_seed_ie = gr.Slider(0, 1000000, 0, step=1, label="Seed")
                    edit_cfg_text_scale_ie = gr.Slider(1.0, 8.0, 4.0, step=0.1, label="CFG Text Scale")
                with gr.Row():
                    edit_cfg_img_scale_ie = gr.Slider(1.0, 4.0, 2.0, step=0.1, label="CFG Image Scale")
                    edit_cfg_interval_ie = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Interval Start")
                with gr.Row():
                    edit_cfg_renorm_type_ie = gr.Dropdown(["global", "local", "text_channel"], value="text_channel", label="CFG Renorm Type")
                    edit_cfg_renorm_min_ie = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Renorm Min")
                with gr.Row():
                    edit_num_timesteps_ie = gr.Slider(10, 100, 50, step=5, label="Timesteps")
                    edit_timestep_shift_ie = gr.Slider(1.0, 10.0, 3.0, step=0.5, label="Timestep Shift")
                
                edit_thinking_params_ie = gr.Group(visible=False)
                with edit_thinking_params_ie:
                    with gr.Row():
                        edit_do_sample_ie = gr.Checkbox(label="Sampling (Text)", value=False)
                        edit_max_think_token_n_ie = gr.Slider(64, 4006, 1024, step=64, label="Max Think Tokens")
                        edit_text_temperature_ie = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Temperature (Text)")
        
        edit_btn_ie = gr.Button("Edit Image")
        
        def ui_update_edit_thinking_visibility(show: bool):
            return gr.update(visible=show), gr.update(visible=show)
        
        edit_show_thinking_ie.change(
            fn=ui_update_edit_thinking_visibility, inputs=[edit_show_thinking_ie],
            outputs=[edit_thinking_output_ie, edit_thinking_params_ie]
        )
        
        def process_edit_request(
            image, prompt, show_thinking, cfg_text_scale, cfg_img_scale, cfg_interval, 
            timestep_shift, num_timesteps, cfg_renorm_min, cfg_renorm_type, 
            max_think_token_n, do_sample, text_temperature, seed_val
        ):
            edited_img, thinking_text = edit_image( # Calls global edit_image
                image, prompt, show_thinking, cfg_text_scale, cfg_img_scale, 
                cfg_interval, timestep_shift, num_timesteps, cfg_renorm_min, 
                cfg_renorm_type, max_think_token_n, do_sample, text_temperature, seed_val
            )
            return edited_img, thinking_text if thinking_text else ""
        
        edit_btn_ie.click(
            fn=process_edit_request,
            inputs=[
                edit_image_input_ie, edit_prompt_ie, edit_show_thinking_ie, 
                edit_cfg_text_scale_ie, edit_cfg_img_scale_ie, edit_cfg_interval_ie,
                edit_timestep_shift_ie, edit_num_timesteps_ie, edit_cfg_renorm_min_ie, 
                edit_cfg_renorm_type_ie, edit_max_think_token_n_ie, 
                edit_do_sample_ie, edit_text_temperature_ie, edit_seed_ie
            ],
            outputs=[edit_image_output_ie, edit_thinking_output_ie]
        )
    return edit_image_input_ie


def create_image_understanding_tab():
    with gr.Tab("🖼️ Image Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input_iu = gr.Image(
                    label="Input Image", value=load_example_image('test_images/meme.jpg')
                )
                understand_prompt_iu = gr.Textbox(
                    label="Question/Prompt", 
                    value="Can someone explain what's funny about this meme??"
                )
            with gr.Column(scale=1):
                txt_output_iu = gr.Textbox(label="Answer", lines=10) # Increased lines slightly
        
        with gr.Row():
            understand_show_thinking_iu = gr.Checkbox(label="Thinking", value=False)
        
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Row():
                understand_do_sample_iu = gr.Checkbox(label="Sampling", value=False)
                understand_text_temperature_iu = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Temperature")
                understand_max_new_tokens_iu = gr.Slider(64, 4096, 512, step=64, label="Max New Tokens")
        
        img_understand_btn_iu = gr.Button("Understand Image")
        
        # For this tab, thinking process is not explicitly shown in a separate box,
        # but the image_understanding function itself might include it in its main output.
        # If a separate box is needed, it would be similar to other tabs.

        def process_understanding_request(
            image, prompt, show_thinking, do_sample, text_temperature, max_new_tokens
        ):
            return image_understanding( # Calls global image_understanding
                image, prompt, show_thinking, do_sample, text_temperature, max_new_tokens
            )
        
        img_understand_btn_iu.click(
            fn=process_understanding_request,
            inputs=[
                img_input_iu, understand_prompt_iu, understand_show_thinking_iu,
                understand_do_sample_iu, understand_text_temperature_iu, 
                understand_max_new_tokens_iu
            ],
            outputs=txt_output_iu
        )
    return img_input_iu


# --- Main Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("""
    <div>
      <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="380"/>
    </div>
    """)
    
    model_status_textbox = gr.Textbox(
        label="Model Status", value=initial_load_status, interactive=False
    )
    
    dropdown_choices = [m["name"] for m in available_models_list] if available_models_list else []
    initial_dropdown_value = current_model_display_name if current_model_display_name else \
                             (dropdown_choices[0] if dropdown_choices else None)

    model_selector = gr.Dropdown(
        choices=dropdown_choices,
        value=initial_dropdown_value,
        label="Select Model",
        info="Choose a model to load for inference."
    )

    def handle_model_change(selected_display_name: str) -> str:
        selected_model_id = next(
            (m["id"] for m in available_models_list if m["name"] == selected_display_name), None
        )
        if selected_model_id:
            status = reload_model_and_inferencer(selected_model_id)
            # Potentially update other UI elements if needed upon model change
            return status
        return "Failed to find model ID for selection. Please check model registry."

    model_selector.change(fn=handle_model_change, inputs=[model_selector], outputs=[model_status_textbox])

    # Create and populate tabs using the new functions
    create_text_to_image_tab()
    create_image_edit_tab()
    create_image_understanding_tab()

    gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://discord.gg/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</div>
""")

demo.launch(share=True)