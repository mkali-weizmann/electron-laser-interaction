import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

print(os.getcwd())
# Open the image form working directory
image = Image.open("Data Arrays/Static Data/Letters Different Sizes.png")
image_np = np.array(image)
# %%
N = 128
m = 2
n = m * N
shift_x = 0
shift_y = 0
image_np_cropped = image_np[shift_x : n + shift_x : m, shift_y : n + shift_y : m, 0]
plt.imshow(image_np_cropped, cmap="gray")
plt.colorbar()
plt.show()
np.save(f"Data Arrays/Static Data/letters_{N}.npy", image_np_cropped)

# %%
N = 256
m = 2
n = m * N
shift_x = 0
shift_y = 0
image_np_cropped = image_np[shift_x : n + shift_x : m, shift_y : n + shift_y : m, 0]
plt.imshow(image_np_cropped, cmap="gray")
plt.colorbar()
plt.show()
np.save(f"Data Arrays/Static Data/letters_{N}.npy", image_np_cropped)


# %%
N = 64
m = 4
n = m * N
shift_x = 0
shift_y = 0
image_np_cropped = image_np[-(n + shift_x) : -shift_x - 1 : m, -(n + shift_y) : -shift_y - 1 : m, 0]
plt.imshow(image_np_cropped, cmap="gray")
plt.colorbar()
plt.show()
np.save(f"Data Arrays/Static Data/letters_{N}.npy", image_np_cropped)
