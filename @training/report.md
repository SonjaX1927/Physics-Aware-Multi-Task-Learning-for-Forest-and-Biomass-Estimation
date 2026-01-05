## Summary


\ L_\text{total} = L_\text{tasks} + \lambda_\text{allom} , L_\text{allom}

### Genus segmentation

| Experiment | Key configuration | Pixel accuracy | Mean IoU |
| --- | --- | --- | --- |
| baseline-unet-seg | base_channels=256, max_lr=1e-4, weight_decay=1e-4, patience=8, dropout=0.3, epochs=100 | 0.5025 | 0.0647 |

### Height regression

| Experiment | Key configuration | RMSE | R² |
| --- | --- | --- | --- |
| baseline-unet-height | base_channels=256, max_lr=1e-4, weight_decay=1e-4, patience=8, dropout=0.3, epochs=100 | 5.3346 | 0.1576 |

### Biomass regression

| Experiment | Key configuration | RMSE | R² |
| --- | --- | --- | --- |
| baseline-unet-biomass | base_channels=256, max_lr=5e-4, weight_decay=1e-5, patience=4, dropout=0.2, epochs=100 | 42.3494 | 0.0808 |

## Experiments

### baseline-unet-seg

- **Configuration**
  - base_channels: 256
  - max_lr: 1e-4
  - weight_decay: 1e-4
  - patience: 8
  - dropout: 0.3
  - epochs: 100

- **Results**
  - loss: 1.4544
  - pixel_acc: 0.5025
  - mean_iou: 0.0647

- **Result plots**

  <img src="../@plots/training-results/baseline-unet/seg-curve.png" alt="Segmentation training and validation loss" width="40%">
  <p>
    <img src="../@plots/training-results/baseline-unet/seg_cm.png" alt="Segmentation confusion matrix" width="40%">
    <img src="../@plots/training-results/baseline-unet/seg_sample.png" alt="Segmentation prediction samples" width="28%">
  </p>

### baseline-unet-height

- **Configuration**
  - base_channels: 256
  - max_lr: 1e-4
  - weight_decay: 1e-4
  - patience: 8
  - dropout: 0.3
  - epochs: 100

- **Results**
  - loss: 5.3357
  - height_rmse: 5.3346
  - height_r2: 0.1576

- **Result plots**

  <img src="../@plots/training-results/baseline-unet/height-curve.png" alt="Height regression training and validation loss" width="40%">
  <p>
    <img src="../@plots/training-results/baseline-unet/height-sample.png" alt="Height prediction samples" width="40%">
    <img src="../@plots/training-results/baseline-unet/height-scatter.png" alt="Predicted vs. true height scatter" width="26%">
  </p>

### baseline-unet-biomass

- **Configuration**
  - base_channels: 256
  - max_lr: 5e-4
  - weight_decay: 1e-5
  - patience: 4
  - dropout: 0.2
  - epochs: 100

- **Results**
  - loss: 41.6485
  - biomass_rmse: 42.3494
  - biomass_r2: 0.0808

- **Result plots**

  <img src="../@plots/training-results/baseline-unet/biomass-curve.png" alt="Biomass regression training and validation loss" width="40%">
  <p>
    <img src="../@plots/training-results/baseline-unet/biomass-sample.png" alt="Biomass prediction samples" width="40%">
    <img src="../@plots/training-results/baseline-unet/biomass-scatter.png" alt="Predicted vs. true biomass scatter" width="26%">
  </p>