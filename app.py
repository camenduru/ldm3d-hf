from diffusers import StableDiffusionLDM3DPipeline
import gradio as gr
import torch
from PIL import Image
import base64
from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path

Path("tmp").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionLDM3DPipeline.from_pretrained(
    "Intel/ldm3d-4c",
    torch_dtype=torch.float16
    # , safety_checker=None
)
pipe.to(device)
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


def get_iframe(rgb_path: str, depth_path: str, viewer_mode: str = "6DOF"):
    # buffered = BytesIO()
    # rgb.convert("RGB").save(buffered, format="JPEG")
    # rgb_base64 = base64.b64encode(buffered.getvalue())
    # buffered = BytesIO()
    # depth.convert("RGB").save(buffered, format="JPEG")
    # depth_base64 = base64.b64encode(buffered.getvalue())

    # rgb_base64 = "data:image/jpeg;base64," + rgb_base64.decode("utf-8")
    # depth_base64 = "data:image/jpeg;base64," + depth_base64.decode("utf-8")
    rgb_base64 = f"/file={rgb_path}"
    depth_base64 = f"/file={depth_path}"
    if viewer_mode == "6DOF":
        return f"""<iframe src="file=static/three6dof.html" width="100%" height="500px" data-rgb="{rgb_base64}" data-depth="{depth_base64}"></iframe>"""
    else:
        return f"""<iframe src="file=static/depthmap.html" width="100%" height="500px" data-rgb="{rgb_base64}" data-depth="{depth_base64}"></iframe>"""


def predict(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 5.0,
    seed: int = 0,
    randomize_seed: bool = True,
):
    generator = torch.Generator() if randomize_seed else torch.manual_seed(seed)
    output = pipe(
        prompt,
        width=1024,
        height=512,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=40,
    )  # type: ignore
    rgb_image, depth_image = output.rgb[0], output.depth[0]  # type: ignore
    with NamedTemporaryFile(suffix=".png", delete=False, dir="tmp") as rgb_file:
        rgb_image.save(rgb_file.name)
        rgb_image = rgb_file.name
    with NamedTemporaryFile(suffix=".png",  delete=False,  dir="tmp") as depth_file:
        depth_image.save(depth_file.name)
        depth_image = depth_file.name

    iframe = get_iframe(rgb_image, depth_image)
    return rgb_image, depth_image, generator.seed(), iframe


with gr.Blocks() as block:
    gr.Markdown(
        """
## LDM3d Demo 

model: https://huggingface.co/Intel/ldm3d<br>
[Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/ldm3d_diffusion)

"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=10, step=0.1, value=5.0
            )
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            seed = gr.Slider(label="Seed", minimum=0,
                             maximum=2**64 - 1, step=1)
            generated_seed = gr.Number(label="Generated Seed")
            markdown = gr.Markdown(label="Output Box")
            with gr.Row():
                new_btn = gr.Button("New Image")
        with gr.Column(scale=2):
            html = gr.HTML()
            with gr.Row():
                rgb = gr.Image(label="RGB Image", type="filepath")
                depth = gr.Image(label="Depth Image", type="filepath")
    gr.Examples(
        examples=[
            ["A picture of some lemons on a table panoramic view", "", 5.0, 0, True]],
        inputs=[prompt, negative_prompt, guidance_scale, seed, randomize_seed],
        outputs=[rgb, depth, generated_seed, html],
        fn=predict,
        cache_examples=True)

    new_btn.click(
        fn=predict,
        inputs=[prompt, negative_prompt, guidance_scale, seed, randomize_seed],
        outputs=[rgb, depth, generated_seed, html],
    )

block.launch()
