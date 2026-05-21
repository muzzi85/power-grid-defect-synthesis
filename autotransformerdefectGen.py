from GenLib import models_upload, mask_gen, defect_gen
import matplotlib.pyplot as plt

## load models 
model_path = "/Image2image generation/Anomaly detection images/sam2_b.pt"
modelSAM, pipe = models_upload(model_path)

## generate masks
image_path = "/Image2image generation/Anomaly detection images/2d/EPRI_Data_cropped/transformers/1 (4)_transformers_1.jpg"
b_mask, img, sub_mask = mask_gen(modelSAM, image_path)

## generate defective image
prompt = "crack"
guidance_scale_factor = 0.008
num_inference_steps_factor = 100
strength_factor = 1
generated_image = defect_gen(prompt, pipe, img,b_mask, guidance_scale_factor ,num_inference_steps_factor, strength_factor)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
ax1.imshow(generated_image)
ax1.set_title("defects generated")
ax2.imshow(img)
ax2.set_title("Original image")
plt.show()


