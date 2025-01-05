from inference import SwinDRNetPipeline
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from testdata.log.save_log import Save
# test
model_path = "models/model.pth"
rgb = np.array(Image.open("testdata/000200-color.png").convert('RGB'))
depth = np.array(cv2.imread("testdata/000200-depth.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
true = np.array(cv2.imread("testdata/000200-depth_true.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
ppl = SwinDRNetPipeline(model_path)
result = ppl.inference(rgb, depth)

# show
rgb_scaled = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
depth_scaled = cv2.applyColorMap((depth/10).astype(np.uint8), cv2.COLORMAP_TURBO)
result_scaled = cv2.applyColorMap((result/10).astype(np.uint8), cv2.COLORMAP_TURBO)
true_scaled = cv2.applyColorMap((true/10).astype(np.uint8), cv2.COLORMAP_TURBO)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes[0, 0].imshow(rgb_scaled, cmap='gray')
axes[0, 0].set_title('Scaled RGB Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(depth_scaled, cmap='gray')
axes[0, 1].set_title('Scaled Depth Image')
axes[0, 1].axis('off')

axes[1, 0].imshow(true_scaled, cmap='gray')
axes[1, 0].set_title('Scaled True Depth Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(result_scaled, cmap='gray')
axes[1, 1].set_title('Scaled Repaired Depth Image')
axes[1, 1].axis('off')

plt.tight_layout()

canvas = FigureCanvas(fig)
canvas.draw()
canvas_image = np.array(canvas.renderer.buffer_rgba())
canvas_image_pil = Image.fromarray(canvas_image)
Save({'result': canvas_image_pil})
canvas_image_pil.show()