import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# config
IMG_SIZE   = (512, 512)       # your cfg.img_size
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# label mapping
id_to_label = {
    0:"road",         1:"sidewalk",    2:"building",      3:"wall",
    4:"fence",        5:"pole",        6:"traffic light", 7:"traffic sign",
    8:"vegetation",   9:"terrain",    10:"sky",           11:"person",
    12:"rider",      13:"car",        14:"truck",         15:"bus",
    16:"train",      17:"motorcycle", 18:"bicycle",
}
label_to_id = {v: k for k, v in id_to_label.items()}


PALETTE = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
    (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
    (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
    (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
    (  0, 80,100), (  0,  0,230), (119, 11, 32),
]

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),          # same as val_transforms
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def mask_to_colour(mask: np.ndarray) -> np.ndarray:
    colour = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(PALETTE):
        colour[mask == cls_id] = rgb
    colour[mask == 255] = (0, 0, 0)
    return colour


def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=19,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load("best.pth", map_location=DEVICE)
    state_dict = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()
model = load_model()


def run_inference(pil_image: Image.Image):
    img_tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast(DEVICE):
        out    = model(pixel_values=img_tensor)
        logits = F.interpolate(out.logits, size=IMG_SIZE,
                               mode="bilinear", align_corners=False)

    pred        = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    mask_colour = Image.fromarray(mask_to_colour(pred))
    return mask_colour  

def make_legend() -> Image.Image:
    patches = [
        mpatches.Patch(color=[c/255 for c in PALETTE[i]],
                       label=id_to_label[i])
        for i in range(19)
    ]
    fig, ax = plt.subplots(figsize=(14, 1.2))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", ncol=10, fontsize=8, frameon=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()

legend_img = make_legend()


DATASET_RESULTS = [
    ("result/idx_001.png",  "Val #001"),
    ("result/idx_003.png",  "Val #003"),
    ("result/idx_120.png",  "Val #120"),
    ("result/idx_123.png",  "Val #123"),
    ("result/idx_300.png",  "Val #300")
]

EXAMPLES = [
    [os.path.join(os.path.dirname(__file__), "example/example_1.jpg")],
    [os.path.join(os.path.dirname(__file__), "example/example_2.jpg")],
    [os.path.join(os.path.dirname(__file__), "example/example_3.png")],
    [os.path.join(os.path.dirname(__file__), "example/example_4.jpeg")],
    [os.path.join(os.path.dirname(__file__), "example/example_5.jpg")],
]
# Gradio UI 
with gr.Blocks(title="Cityscapes Segmentation") as demo:

    gr.Markdown(
        "# 🏙️ Cityscapes Semantic Segmentation\n"
        "**Model:** SegFormer-B2 · **Training:** 80 epochs · "
        "**Backbone LR:** 6e-5 · **Loss:** 0.7 CE + 0.3 Dice"
    )

    with gr.Tab("Try it yourself"):

        # ── Row 1: Upload left, Output right ───────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload a street image",
                                    height=512,
                                    width=512)
                run_btn   = gr.Button("Run ▶", variant="primary")
            with gr.Column(scale=1):
                out_mask = gr.Image(label="Prediction mask",
                    interactive=False,
                    height=512,
                    width=512)

        # ── Row 2: Examples under the button ───────────────────────
        with gr.Row():
            gr.Examples(
                examples=EXAMPLES,
                inputs=input_img,
                outputs=[out_mask],
                fn=run_inference,
                cache_examples=False,
            )

        gr.Image(value=legend_img, label="Colour legend — 19 Cityscapes classes",
                interactive=False)

        run_btn.click(fn=run_inference,
                    inputs=input_img,
                    outputs=[out_mask])

    # ── Tab 2: Dataset results gallery ────────────────────────────
    with gr.Tab("Dataset results"):
        gr.Markdown(
            "Pre-computed results on the **Cityscapes validation set**.\n"
            "Each row shows: raw image · ground truth · model prediction."
        )
        for path, caption in DATASET_RESULTS:
            gr.Image(value=path, label=caption, interactive=False)

demo.launch()