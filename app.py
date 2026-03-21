import gradio as gr
import torch
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device)

tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
tokenizer.pad_token = tokenizer.eos_token
decoder = GPT2LMHeadModel.from_pretrained("./decoder").to(device)
decoder.eval()

def analyze_artifact(image):
    try:
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        visual_probes = {
            "shape": [
                "a mummiform or wrapped shape",
                "a beetle or oval shape",
                "a human or animal figurine shape",
                "a flat rectangular plaque shape",
                "a cylindrical or vessel shape",
                "a box or container shape"
            ],
            "material": [
                "made of blue or green faience glaze",
                "made of dark bronze or copper metal",
                "made of white or grey limestone",
                "made of gold or gilded material",
                "made of red or brown terracotta clay",
                "made of black basalt stone"
            ],
            "surface": [
                "with hieroglyphic inscriptions or text",
                "with carved relief decoration",
                "with painted decoration",
                "with smooth undecorated surface",
                "with animal or deity iconography"
            ]
        }

        observations = {}
        for feature, probes in visual_probes.items():
            tokens = clip.tokenize(probes).to(device)
            with torch.no_grad():
                img_feat  = model.encode_image(img_tensor)
                txt_feat  = model.encode_text(tokens)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            top_idx = sim[0].argmax().item()
            observations[feature] = probes[top_idx]

        prompt = (
            f"Examine this Egyptian artifact and explain your reasoning.\n\n"
            f"[Observation]\n"
            f"- Shape:    {observations['shape']}\n"
            f"- Material: {observations['material']}\n"
            f"- Surface:  {observations['surface']}\n\n"
            f"[Reasoning]\n"
        )

        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = decoder.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=analyze_artifact,
    inputs=gr.Image(type="pil", label="Upload Egyptian Artifact Image"),
    outputs=gr.Textbox(label="Model Reasoning", lines=15),
    title="🏺 Egyptian_Artifact_VLM",
    description="""
**Vision-Language Reasoning for Ancient Egyptian Artifacts (700–330 BCE)**

Upload an image of a Late Period Egyptian artifact.
The model reasons about what it is — not just a label, but *why*,
citing visible features like shape, material, and surface decoration.

Built with CLIP + distilgpt2 | British Museum Collection | Late Period Egypt
    """,
    theme=gr.themes.Soft()
)

demo.launch()
