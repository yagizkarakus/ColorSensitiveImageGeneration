from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Union
import io
from PIL import Image, ImageDraw, ImageOps, ImageColor,ImageFont
import textwrap
from fastapi.responses import StreamingResponse
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image, ImageColor
from transformers import pipeline
import ngrok
import nest_asyncio
import uvicorn
from dotenv import load_dotenv
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add the origin of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#This piece of code will be executed only when the server starts
@app.on_event("startup")
def on_startup():
    #Don't forget to declare pipe as a global variable
    global img2img_pipe
    global sam
    global depth_estimator
    global controlnet
    global contolnet_pipe


    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    img2img_pipe = img2img_pipe.to("mps")
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="/Users/yagito/Desktop/TestControlNet/SAM_MODEL_CHECKPOINT/sam_vit_h_4b8939.pth")
    sam.to(device="cpu")
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float32),
        ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float32),
    ]
    contolnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
    )



@app.get("/")
async def HelloWorld() -> dict:
    return {"message":"Hello AdCreative AI"}


@app.post("/ColorSensitiveGeneration/")
async def ColorSensitiveGeneration(
    image_file: UploadFile = File(...),
    prompt: str = Form(...),
    Hexcolor: str = Form(...) ):
    
    generator = torch.Generator(device="cpu").manual_seed(1)

    # Image2Image Pipe
    img2img_pipe.enable_attention_slicing()
    
    image_content = await image_file.read()
    image = Image.open(io.BytesIO(image_content))
    first_image = img2img_pipe(prompt, image=image, num_inference_steps=25, generator=generator).images[0]


    # Segemnt generated Image and paint for contolnet
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_rgb = np.array(first_image)
    result = mask_generator.generate(image_rgb)

    masks = [
        mask['segmentation']
        for mask in sorted(result, key=lambda x: x['area'], reverse=True)
    ]

    original_image = first_image.copy()
    new_color = ImageColor.getcolor(Hexcolor, "RGB")

    mask_array = np.array(masks[0], dtype=np.uint8)

    original_image_array = np.array(original_image)

    original_image_array[mask_array != 0] = new_color

    result_image = Image.fromarray(original_image_array)
    
    
    gray_image = original_image.convert('L')
    
    np_image = np.array(gray_image)
    # get canny image
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    #get depth image
    depth_image = depth_estimator(images=result_image)
    depth_image_pillow = depth_image['depth']

    depth_image_np_normalized = (np.array(depth_image_pillow) - np.array(depth_image_pillow).min()) / (np.array(depth_image_pillow).max() - np.array(depth_image_pillow).min())

    depth_pillow_image = Image.fromarray((depth_image_np_normalized * 255).astype(np.uint8))

    controlimages = [canny_image, depth_pillow_image]
    #contolnet pipeline 
    
    contolnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(contolnet_pipe.scheduler.config)
    contolnet_pipe.enable_attention_slicing()
    
    final_image = contolnet_pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=result_image,
        control_image=controlimages,
        guidance_scale=10,
        controlnet_conditioning_scale=[0.8, 1.0]
    ).images[0]


    result_bytesio = io.BytesIO()
    final_image.save(result_bytesio, format="PNG")

    # Set the BytesIO object position to the beginning
    result_bytesio.seek(0)

    return StreamingResponse(result_bytesio, media_type="image/png", headers={"Content-Disposition": "attachment;filename=ColorSensitiveGenerationResult.png"})






@app.post("/DynamicAdImage/")
async def DynamicAdImage(
    image_file: UploadFile = File(...),
    logo_file: UploadFile = File(...),
    Punchline: str = Form(...),
    ctaText: str = Form(...),
    Hexcolor: str = Form(...)
):

    color = ImageColor.getcolor(Hexcolor, "RGB")
    

    canvas_width = 1280
    canvas_height = 1280

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    
    logo_content = await logo_file.read()
    logo = Image.open(io.BytesIO(logo_content))

    image_content = await image_file.read()
    image = Image.open(io.BytesIO(image_content))
    image = ImageOps.exif_transpose(image)

    logo_width, logo_height = logo.size
    logo_margin = 80

    desired_logo_height = 100

    max_logo_height = desired_logo_height
    max_logo_width = (max_logo_height / logo_height) * logo_width
    logo.thumbnail((max_logo_width, max_logo_height))

    adjusted_logo_position = (int((canvas_width - max_logo_width) // 2), int(logo_margin))

    canvas.paste(logo, adjusted_logo_position, logo.convert("RGBA"))


    image_margin_top = 50
    image_width, image_height = image.size
    max_image_height = 600  
    max_image_width = (max_image_height / image_height) * image_width
    image.thumbnail((max_image_width, max_image_height))

    image_position = (
        int((canvas_width - image.width) // 2),
        logo.height+logo_margin+image_margin_top 
    )

    mask = Image.new("RGBA", (image.width, image.height), (255, 255, 255, 0))
    mask_draw = ImageDraw.Draw(mask,"RGBA")
    mask_draw.rounded_rectangle((0, 0, image.width, image.height), radius=20, fill=(0,255,255,255))


    canvas.paste(image, image_position, mask)


    caption = Punchline
    wrapper = textwrap.TextWrapper(width=int(canvas_width * 0.03))
    word_list = wrapper.wrap(text=caption)
    caption_new = "\n".join(word_list)

    text_draw = ImageDraw.Draw(canvas)
    fnt = ImageFont.truetype("Arial Bold.ttf", 70)

    center_x = canvas_width // 2

    text_width, text_height = text_draw.multiline_textsize(caption_new, font=fnt)
    text_x = center_x - text_width // 2

    # Draw the multiline text horizontally centered
    text_draw.multiline_text((text_x, image_position[1]+image.height+ 50), caption_new, font=fnt, fill=color, align='center')



    button_caption = ctaText
    button_caption_wrapper = textwrap.TextWrapper(width=25)
    button_word_list = button_caption_wrapper.wrap(text=button_caption)
    button_caption_new = "\n".join(button_word_list)

    button_draw = ImageDraw.Draw(canvas)
    btn_fnt = ImageFont.truetype("Arial Bold.ttf", 25)

    cta_text_width, cta_text_height = button_draw.multiline_textsize(button_caption_new, font=btn_fnt)




    button_position = (
        int((canvas_width - cta_text_width ) // 2),
        int(image_position[1] + max_image_height+ text_height +20),
    )

    button_draw.rounded_rectangle(
        [button_position[0]-20, button_position[1]-20, button_position[0] + cta_text_width+20, button_position[1] + cta_text_height+20],
        fill=color,
        width=50,
        radius=17,
    )




    button_caption_position = (
        button_position[0],
        button_position[1],  
    )

    button_draw.multiline_text(button_caption_position, button_caption_new, font=btn_fnt, fill=(255, 255, 255), align="center")


    button_draw.rounded_rectangle((canvas_width//15, -15 , canvas_width-(canvas_width//15),  15 ), fill=color, radius=20)

    button_draw.rounded_rectangle((canvas_width//15, canvas_height-15 , canvas_width-(canvas_width//15),  canvas_height+15 ), fill=color, radius=20)

    result_bytesio = io.BytesIO()
    canvas.save(result_bytesio, format="PNG")

    # Set the BytesIO object position to the beginning
    result_bytesio.seek(0)

    return StreamingResponse(result_bytesio, media_type="image/png", headers={"Content-Disposition": "attachment;filename=DynamicAdImageResult.png"})

load_dotenv()

token = os.getenv("authtoken")
ngrok.set_auth_token(token)
ngrok_tunnel = ngrok.forward(8000, authtoken_from_env=True)

public_url = ngrok_tunnel.url()
print(100*"*")
print('Public URL: ', public_url+"/docs")
print(100*"*")
nest_asyncio.apply()
uvicorn.run(app, port=8000)