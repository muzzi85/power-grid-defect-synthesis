import sys
sys.path.append("/Image2image generation/Anomaly detection images")
from ultralytics import YOLO
from ultralytics import SAM
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops 
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from defectGenerator import DefectGenerator
from matplotlib import cm
from styleTransfer import style_transfer_test


def models_upload(model_path):

    ## inpaiting model 

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

    ## run SAM to detect objects

    # Load a model
    #modelSAM = SAM("/home/muzzi/Image2image generation/Anomaly detection images/sam2_b.pt")
    modelSAM = SAM(model_path)

    # Display model information (optional)
    modelSAM.info()

    return modelSAM, pipe


def mask_gen(modelSAM, image_path):
    ## test 1 image 
    #image_path = "/home/muzzi/Image2image generation/Anomaly detection images/2d/EPRI_Data_cropped/transformers/1 (4)_transformers_1.jpg"
    #image_path = "/home/muzzi/Image2image generation/Anomaly detection images/2d/EPRI_Data_cropped/transformers/1 (4)_transformers_1.jpg"

    image = load_image(image_path).convert('RGB')
    image_size_w = image.size[0]
    image_size_h = image.size[1]

    prompt_points = [int(np.round(image_size_w/2)), int(np.round(image_size_h/2))]
    # Segment with bounding box prompt
    results = modelSAM(image_path, save=True, conf=0.1, points = prompt_points) #, bboxes=[100, 100, 200, 200], points=[100, 375]

    # Display results
    for result in results:
        result.show()

    #  Iterate detection results (helpful for multiple images)
    for r in results:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem  # source image base-name

        # Iterate each object contour (multiple detections)
        for ci, c in enumerate(r[0]):
            #  Get detection class name
            label = c.names[c.boxes.cls.tolist().pop()]

    # Create binary mask
    b_mask = np.zeros(img.shape[:2], np.uint8)

    #  Extract contour result
    contour = c.masks.xy.pop()
    #  Changing the type
    contour = contour.astype(np.int32)
    #  Reshaping
    contour = contour.reshape(-1, 1, 2)
    # Draw contour onto mask
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    ax1.imshow(b_mask)
    ax2.imshow(img)
    plt.show()

    ## sub mask
    mask_size_w = b_mask.shape[0]
    mask_size_h = b_mask.shape[1]

    remove_rate_w = int(mask_size_w/1.2)
    remove_rate_h = int(mask_size_h/1.2)

    b_mask[0:remove_rate_h] = 0 
    b_mask[remove_rate_w:] = 0 

    sub_mask = np.zeros((b_mask.shape[0],b_mask.shape[1]))
    sub_mask [400:600, 700:800]= b_mask[400:600, 700:800]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 6))
    ax1.imshow(b_mask)
    ax1.set_title("mask")
    ax2.imshow(sub_mask)
    ax2.set_title("submask")
    ax3.imshow(image)
    ax3.set_title("Original image")
    plt.show()

    return b_mask, image, sub_mask


def defect_gen(prompt, pipe, image,b_mask, guidance_scale_factor ,num_inference_steps_factor, strength_factor):

    generator = torch.Generator(device="cuda").manual_seed(0)

    imagee = pipe(
    prompt=prompt,
    image=image,
    mask_image=b_mask,
    guidance_scale=guidance_scale_factor, # 0.8
    num_inference_steps=num_inference_steps_factor,  # 20  steps between 15 and 30 work well for us
    strength=strength_factor,  #0.99 make sure to use `strength` below 1.0
    generator=generator,
    ).images[0]

    image_size_w = image.size[0]
    image_size_h = image.size[1]
    generated_image = imagee.resize((image_size_w, image_size_h), Image.LANCZOS)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    ax1.imshow(generated_image)
    ax1.set_title("defects generated")
    ax2.imshow(image)
    ax2.set_title("Original image")
    plt.show()

    return generated_image
