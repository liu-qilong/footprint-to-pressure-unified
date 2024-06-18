import numpy as np
from PIL import Image, ImageOps

def load_foot_mask(self, l_mask_path: str):
    # load foot masks
    l_img = Image.open(l_mask_path)
    r_img = ImageOps.mirror(l_img)

    self.l_mask = np.array(l_img).astype(np.float64)
    self.r_mask = np.array(r_img).astype(np.float64)

    # detect pixels of area no.1~197 and store the corresponding indexes
    self.l_index = {}
    self.r_index = {}

    for n in range(0, 99):
        self.l_index[n] = np.where(self.l_mask == n + 1)
        self.r_index[n + 99] = np.where(self.r_mask == n + 1)

    # index for slicing footprint image as sensor stacks
    range_half = int(self.stack_range / 2)

    self.x_stack_slice = {'L': [], 'R': []}
    self.y_stack_slice = {'L': [], 'R': []}

    for sensor in range(99):
        x_center, y_center = int(self.l_index[sensor][0].mean()), int(self.l_index[sensor][1].mean())
        xs = np.arange(x_center - range_half, x_center + range_half)
        ys = np.arange(y_center - range_half, y_center + range_half)
        xg, yg = np.meshgrid(xs, ys, indexing='ij')
        self.x_stack_slice['L'].append(xg)
        self.y_stack_slice['L'].append(yg)

    for sensor in range(99, 198):
        x_center, y_center = int(self.r_index[sensor][0].mean()), int(self.r_index[sensor][1].mean())
        xs = np.arange(x_center - range_half, x_center + range_half)
        ys = np.arange(y_center - range_half, y_center + range_half)
        xg, yg = np.meshgrid(xs, ys, indexing='ij')
        self.x_stack_slice['R'].append(xg)
        self.y_stack_slice['R'].append(yg)

    self.x_stack_slice['L'] = np.array(self.x_stack_slice['L'])
    self.x_stack_slice['R'] = np.array(self.x_stack_slice['R'])
    self.y_stack_slice['L'] = np.array(self.y_stack_slice['L'])
    self.y_stack_slice['R'] = np.array(self.y_stack_slice['R'])