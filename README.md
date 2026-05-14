# PLS Loss (Positive-guided Local Supervision)
Official PyTorch implementation of PLS loss for robust road extraction, including BCE + Dice baseline.

## Datasets
We provide two annotated datasets for research use.

- CH4P
  Link: https://pan.baidu.com/s/1TjsXzd_3P9hIcLhdVP6PSg
  Password: mpu5

- DG-mini
  Link: https://pan.baidu.com/s/17OSnhCj8DyQH3kR_CKZCrg
  Password: q49j

The CH4P dataset contains only annotation masks. Satellite images can be downloaded using the center coordinates in filenames via the Mapbox API.

## Usage of PLS Loss
The PLS loss (local_dice_bce_loss) is a positive-guided supervision loss that randomly crops local patches around road pixels to enhance training stability and road integrity.

### Example: K=16, S=256
```python
import torch
from losses import local_dice_bce_loss

# Initialize PLS loss with K=16 regions, crop size 256
criterion = local_dice_bce_loss(
    num_regions=16,
    crop_sizes=[256]
)

# Forward
pred = model(input)
loss = criterion(pred, mask)
loss.backward()
