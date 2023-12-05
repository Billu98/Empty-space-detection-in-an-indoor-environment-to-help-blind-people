import torch
# Save raw
from zoedepth.utils.misc import save_raw_16bit
from PIL import Image
from zoedepth.utils.misc import colorize


# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
image = Image.open("test.jpeg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

# depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor
fpath = "output_raw.png"
save_raw_16bit(depth_numpy, fpath)

# Colorize output
colored = colorize(depth_numpy)

# save colored output
fpath_colored = "output_colored.png"
Image.fromarray(colored).save(fpath_colored)
