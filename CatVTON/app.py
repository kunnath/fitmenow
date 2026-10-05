import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(cur_dir, "resource/demo/output"),
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    default_precision = "fp16" if default_device == "mps" else "bf16"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to run model on (cuda, mps, cpu).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=default_precision,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device=args.device
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=args.device, 
)

def submit_function(
    person_image_val,
    person_image_editor_val,
    draw_mask_toggle,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    if draw_mask_toggle:
        person_image = person_image_editor_val
    else:
        person_image = person_image_val

    if person_image is None or cloth_image is None:
        raise gr.Error("Please upload both a Person Image and a Condition Image (or click an example below).")

    if isinstance(person_image, dict):
        bg_img = person_image.get("background")
        layers = person_image.get("layers", [])
        layer_img = layers[0] if (layers and len(layers) > 0) else None
    else:
        bg_img = person_image
        layer_img = None

    if not bg_img:
        raise gr.Error("Please upload or select a Person Image.")

    mask = None
    if layer_img is not None:
        try:
            if isinstance(layer_img, str):
                mask_temp = Image.open(layer_img).convert("L")
            elif isinstance(layer_img, Image.Image):
                mask_temp = layer_img.convert("L")
            elif isinstance(layer_img, np.ndarray):
                mask_temp = Image.fromarray(layer_img).convert("L")
            else:
                mask_temp = None

            if mask_temp is not None:
                mask_arr = np.array(mask_temp)
                if len(np.unique(mask_arr)) > 1:
                    mask_arr[mask_arr > 0] = 255
                    mask = Image.fromarray(mask_arr)
        except Exception:
            mask = None

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        gen_device = 'cpu' if str(args.device).startswith('mps') else args.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    if isinstance(bg_img, str):
        person_image = Image.open(bg_img).convert("RGB")
    elif isinstance(bg_img, Image.Image):
        person_image = bg_img.convert("RGB")
    elif isinstance(bg_img, np.ndarray):
        person_image = Image.fromarray(bg_img).convert("RGB")

    if isinstance(cloth_image, str):
        cloth_image = Image.open(cloth_image).convert("RGB")
    elif isinstance(cloth_image, Image.Image):
        cloth_image = cloth_image.convert("RGB")
    elif isinstance(cloth_image, np.ndarray):
        cloth_image = Image.fromarray(cloth_image).convert("RGB")

    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    try:
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )[0]
    except Exception as e:
        raise gr.Error(
            "An error occurred during inference: {}".format(e)
        )
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image


def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;"> 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="http://arxiv.org/abs/2407.15886" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/CatVTON' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://120.76.142.206:8888" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>
<br>
· This demo and our weights are only for <span>Non-commercial Use</span>. <br>
· You can try CatVTON in our <a href="https://huggingface.co/spaces/zhengchong/CatVTON">HuggingFace Space</a> or our <a href="http://120.76.142.206:8888">online demo</a> (run on 3090). <br>
· Thanks to <a href="https://huggingface.co/zero-gpu-explorers">ZeroGPU</a> for providing A100 for our <a href="https://huggingface.co/spaces/zhengchong/CatVTON">HuggingFace Space</a>. <br>
· SafetyChecker is set to filter NSFW content, but it may block normal results too. Please adjust the <span>`seed`</span> for normal outcomes.<br> 
"""

def app_gradio():
    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    with gr.Column():
                        draw_mask_toggle = gr.Checkbox(
                            label="Draw Mask Manually", value=False
                        )
                        person_image = gr.Image(
                            interactive=True,
                            label="Person Image",
                            type="filepath",
                            sources=["upload", "webcam", "clipboard"],
                            visible=True,
                        )
                        person_image_editor = gr.ImageEditor(
                            interactive=True,
                            label="Draw Mask on Person Image",
                            type="filepath",
                            visible=False,
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )


                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )
                
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=25
                    )
                    # Guidence Scale
                    guidance_scale = gr.Slider(
                        label="CFG Strenth", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                    )
                    # Random Seed
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    cur_script_dir = os.path.dirname(os.path.abspath(__file__))
                    root_path = os.path.join(cur_script_dir, "resource", "demo", "example")

                    def get_example_list(sub_path):
                        target_dir = os.path.join(root_path, *sub_path)
                        if not os.path.exists(target_dir):
                            return []
                        return [
                            os.path.join(target_dir, f)
                            for f in os.listdir(target_dir)
                            if not f.startswith(".")
                        ]

                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=get_example_list(["person", "men"]),
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ①",
                        )
                        women_exm = gr.Examples(
                            examples=get_example_list(["person", "women"]),
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples come from the demos of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>. </span>'
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=get_example_list(["condition", "upper"]),
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=get_example_list(["condition", "overall"]),
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=get_example_list(["condition", "person"]),
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference Person Examples",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>'
                        )

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            def sync_to_editor(img):
                if img is None:
                    return None
                return {"background": img, "layers": [], "composite": None}

            person_image.change(
                sync_to_editor, inputs=person_image, outputs=person_image_editor
            )

            def toggle_mask_editor(visible):
                return (
                    gr.Image(visible=not visible),
                    gr.ImageEditor(visible=visible)
                )

            draw_mask_toggle.change(
                fn=toggle_mask_editor,
                inputs=draw_mask_toggle,
                outputs=[person_image, person_image_editor]
            )

            def stop_fn():
                pipeline._interrupt = True

            click_event = submit.click(
                submit_function,
                [
                    person_image,
                    person_image_editor,
                    draw_mask_toggle,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )
            stop.click(fn=stop_fn, cancels=[click_event])
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
