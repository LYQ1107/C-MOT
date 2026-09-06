# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
# ------------------------------------------------------------------------
import os

import torch
import torch.nn as nn
try:
    from clip import clip
except ImportError:  # Semantic tensors can be loaded without the CLIP package.
    clip = None

from .coco_categories import COCO_CATEGORIES
from .lvis_v1_categories import LVIS_CATEGORIES
import torch.nn.functional as F

def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def load_clip_to_cpu(visual_backbone):
    if clip is None:
        raise ImportError("CLIP package is required only to create new embeddings")
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def load_embeddings(Clip_text_embeddings, Clip_image_embeddings=None, device=None):
    """Load semantic assets without selecting a device as a side effect.

    The upstream implementation unconditionally called ``.cuda()`` while
    constructing a model.  That made CPU preflight unsafe and could silently
    allocate on the caller's default GPU.  C-MOT keeps both assets on CPU
    until the model explicitly moves the selected columns to its runtime
    device.  ``device`` is accepted for callers that want an explicit,
    controlled destination, but no environment-wide device is inferred.
    """
    if Clip_text_embeddings is None:
        raise ValueError("Clip_text_embeddings is required")
    text_embedding = torch.load(Clip_text_embeddings, map_location="cpu")
    image_embedding = None
    if Clip_image_embeddings is not None:
        image_embedding = torch.load(Clip_image_embeddings, map_location="cpu")
    if not isinstance(text_embedding, torch.Tensor):
        raise TypeError("text embedding asset must contain a Tensor")
    if image_embedding is not None and not isinstance(image_embedding, torch.Tensor):
        raise TypeError("image embedding asset must contain a Tensor")
    if device is not None:
        text_embedding = text_embedding.to(device)
        if image_embedding is not None:
            image_embedding = image_embedding.to(device)
    return text_embedding, image_embedding

def build_text_embedding_lvis():
    if clip is None:
        raise ImportError("CLIP package is required only to create new embeddings")
    categories = LVIS_CATEGORIES
    model, _ = clip.load("ViT-B/32")
    templates = multiple_templates

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category["name"], rm_dot=True), article=article(category["name"])
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

    all_text_embeddings = all_text_embeddings.t()
    return all_text_embeddings
