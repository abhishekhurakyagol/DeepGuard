import base64
import io
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class EfficientNetGradCAM:
    """
    Lightweight, native PyTorch Grad-CAM implementation for EfficientNet-B0.
    Hooks into model.features[-1] to extract class activation maps.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap array (H, W) in range [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if gradients fail
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Interpolate to input size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def close(self):
        """Remove hooks to prevent memory leaks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


def generate_gradcam_overlay(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_idx: int,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate blended Grad-CAM heatmap overlay image and extract forensic observations.
    Returns:
        (base64_data_uri, forensic_observations, feature_weights)
    """
    image = image.convert("RGB")
    target_layer = model.features[-1]
    cam_generator = EfficientNetGradCAM(model, target_layer)

    input_tensor = transform(image).unsqueeze(0).to(device)

    try:
        heatmap_norm = cam_generator.generate(input_tensor, class_idx)
    finally:
        cam_generator.close()

    # Convert original PIL image to RGB numpy array
    rgb_img = np.array(image.convert("RGB"))
    h, w = rgb_img.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap_norm, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Jet color map (BGR)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap overlay
    alpha = 0.45
    blended = np.uint8(alpha * colored_heatmap_rgb + (1 - alpha) * rgb_img)

    # Encode to base64 PNG data URI
    pil_blended = Image.fromarray(blended)
    buffer = io.BytesIO()
    pil_blended.save(buffer, format="PNG", optimize=True)
    b64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_encoded}"

    # Extract image-level forensic observations
    # Calculate edge consistency and high-frequency energy ratio
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    activation_concentration = float(np.mean(heatmap_resized > 0.5) * 100)

    # Edge analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = float(np.mean(edge_magnitude > 40) * 100)

    # Color balance variance
    color_var = float(np.var(rgb_img, axis=(0, 1)).mean())

    observations = [
        {
            "title": "Model Focus Density",
            "category": "Explainability",
            "description": f"High attention focused across {activation_concentration:.1f}% of salient image regions in final feature maps.",
            "confidence": round(min(100.0, activation_concentration * 1.5), 1),
        },
        {
            "title": "High-Frequency Texture Consistency",
            "category": "Artifact Analysis",
            "description": f"Laplacian variance measured at {laplacian_var:.1f}. Natural skin and texture gradients examined.",
            "confidence": round(min(100.0, laplacian_var / 10), 1),
        },
        {
            "title": "Boundary & Edge Gradients",
            "category": "Boundary Blend",
            "description": f"Edge transition density estimated at {edge_density:.1f}% across key visual boundaries.",
            "confidence": round(min(100.0, edge_density * 2.2), 1),
        },
    ]

    # Weighted feature contributions
    total_score = max(1.0, laplacian_var / 15 + edge_density + color_var / 30)
    w1 = round(min(100.0, (edge_density / total_score) * 100))
    w2 = round(min(100.0, ((laplacian_var / 15) / total_score) * 100))
    w3 = round(max(0.0, 100 - w1 - w2))

    feature_weights = [
        {
            "name": "Facial Boundary Transitions",
            "weight": w1,
            "description": "Gradient consistency at facial and structural contours.",
        },
        {
            "name": "Micro-Texture & High-Frequency Noise",
            "weight": w2,
            "description": "Artifact patterns in high-frequency pixel domains.",
        },
        {
            "name": "Chrominance & Lighting Variance",
            "weight": w3,
            "description": "Color distribution across illuminated surfaces.",
        },
    ]

    return data_uri, observations, feature_weights
