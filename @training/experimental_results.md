# Experimental Results

This section presents the experimental setup, results, and analysis for both single-task baseline models and the multi-task learning approach for forest attribute estimation from Sentinel-1 SAR imagery.

## Baseline Single-Task Models

### Experimental Setup

Three independent U-Net models were trained for genus segmentation, canopy height regression, and biomass regression. All models used identical training configurations to ensure fair comparison. The training dataset consisted of 724 patches (64×64 pixels) from the spatially blocked training split, with 145 validation patches and 109 test patches held out for evaluation.

**Table 1: Training Hyperparameters for Baseline Models**

| Parameter | Segmentation | Height Regression | Biomass Regression |
|-----------|--------------|-------------------|-------------------|
| Base channels | 128 | 128 | 128 |
| Dropout probability | 0.2 | 0.2 | 0.2 |
| Batch size | 8 | 8 | 8 |
| Maximum learning rate | 3×10⁻⁴ | 3×10⁻⁴ | 3×10⁻⁴ |
| Minimum learning rate | 5×10⁻⁵ | 5×10⁻⁵ | 6×10⁻⁵ |
| Weight decay | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Annealing period | T_max/4 | T_max/3 | T_max/3 |
| Early stopping patience | 10 epochs | 10 epochs | 8 epochs |
| Maximum epochs | 100 | 100 | 100 |
| Optimization metric | Dice coefficient | R² | R² |

The AdamW optimizer was employed for all models, combining the benefits of adaptive learning rates with L2 weight regularization. The cosine annealing learning rate schedule gradually reduced the learning rate from the maximum to minimum value over the specified period, allowing the model to explore the parameter space with larger steps initially before fine-tuning with smaller updates. After reaching the minimum learning rate, training continued at that rate until early stopping criteria were met.

Early stopping monitored the validation metric (Dice coefficient for segmentation, R² for regression) and terminated training if no improvement occurred for the specified patience period. This prevented overfitting while allowing sufficient training time for convergence. The validation metric was evaluated at the end of each epoch on the held-out validation set.

For segmentation, 16 rare genus classes with insufficient training samples were excluded by assigning them an ignore index of -1. These pixels did not contribute to the loss or gradient computation, focusing the model on the six dominant genera (Abies, Fagus, Fraxinus, Picea, Pinus, Quercus) that represent the majority of forest cover. For regression tasks, pixels with invalid targets (NaN or values < -1000) were masked during loss computation, ensuring that only valid forest pixels contributed to training.

### Results

**Table 2: Test Set Performance of Baseline Models**

| Task | Metric | Value |
|------|--------|-------|
| **Segmentation** | Pixel Accuracy | 0.8572 |
| | Mean IoU | 0.2882 |
| | Mean Dice | 0.3657 |
| **Height Regression** | RMSE (m) | 5.243 |
| | R² | 0.255 |
| **Biomass Regression** | RMSE (t/ha) | 39.003 |
| | R² | 0.024 |

The segmentation model achieved 85.7% pixel-level accuracy on the test set, indicating that the majority of pixels were correctly classified. However, the mean IoU of 0.288 and mean Dice coefficient of 0.366 reveal substantial room for improvement in class-wise performance. This discrepancy between pixel accuracy and IoU/Dice metrics reflects class imbalance, where the model performs well on dominant genera but struggles with less common classes.

**Table 3: Per-Genus Segmentation Performance (Test Set)**

| Genus Code | Genus Name | Accuracy | IoU | Dice |
|------------|------------|----------|-----|------|
| 1 | Abies | 0.7637 | 0.4534 | 0.6239 |
| 11 | Fagus | 0.5538 | 0.3408 | 0.5183 |
| 13 | Fraxinus | 0.9312 | 0.0292 | 0.0568 |
| 19 | Picea | 0.8352 | 0.0323 | 0.0626 |
| 20 | Pinus | 0.9251 | 0.8009 | 0.8894 |
| 25 | Quercus | 0.0082 | 0.0076 | 0.0150 |

The per-genus results reveal substantial performance variation across forest types. Pinus achieved the highest Dice score (0.889), likely due to its distinct SAR backscatter signature and good representation in the training data. Abies and Fagus showed moderate performance (Dice: 0.624 and 0.518), while Fraxinus, Picea, and Quercus exhibited poor performance despite high pixel-level accuracy for some classes. The extremely low scores for Quercus (Dice: 0.015) suggest that this genus is either underrepresented or has a SAR signature similar to other genera, making discrimination difficult.

Height regression achieved an R² of 0.255 with RMSE of 5.24 meters. While the model captures some variance in canopy height, the moderate R² indicates that Sentinel-1 C-band backscatter alone provides limited information for precise height estimation. The RMSE of approximately 5 meters is substantial relative to the typical canopy height range (10-35 meters) in the study sites, suggesting that additional features or longer wavelength SAR data may be needed for improved height prediction.

Biomass regression showed the poorest performance with R² of 0.024 and RMSE of 39.0 t/ha. The near-zero R² indicates that the model explains almost none of the variance in biomass, essentially performing no better than predicting the mean biomass value. This result is not surprising given that C-band SAR saturates at relatively low biomass levels (typically 50-100 t/ha) due to limited canopy penetration, while the dataset contains biomass values exceeding 300 t/ha in mature forest stands.

<img src="../@plots/training-results/baseline-unet/baseline-height-scatter-plot.png" width="70%" />

**Figure 7: Height prediction scatter plot for the baseline regression model.** The hexbin density plot shows predicted versus true height values on the test set. The red dashed line represents the fitted regression line, while the black dashed line shows the 1:1 reference. The model captures the general height distribution (evident from the marginal histograms) but exhibits systematic underestimation for tall forests (>30m) and overestimation for shorter forests (<20m). The dense cluster around 20-30m height indicates that the model tends to predict values near the dataset mean, reflecting the limited sensitivity of C-band SAR to canopy height variations.

<img src="../@plots/training-results/baseline-unet/baseline-biomass-scatter-plot.png" width="70%" />

**Figure 8: Biomass prediction scatter plot for the baseline regression model.** The scatter plot reveals severe saturation effects in biomass prediction. The model predictions cluster in a narrow range (50-120 t/ha) regardless of true biomass values, which span 0-400 t/ha. This horizontal banding pattern is characteristic of SAR signal saturation, where backscatter becomes insensitive to biomass increases beyond a threshold. The red regression line deviates substantially from the 1:1 line, confirming that the model cannot capture the full range of biomass variability from C-band backscatter alone.

<img src="../@plots/training-results/baseline-unet/baseline-all-tasks-samples.png" width="100%" />

**Figure 9: Example predictions from baseline models on three test samples.** Each row shows VV and VH backscatter inputs (left), followed by true and predicted outputs for segmentation, biomass, and height. The segmentation predictions (row 1) capture broad spatial patterns but miss fine-scale genus boundaries. Biomass predictions (row 2) show smoothed spatial patterns with reduced dynamic range compared to ground truth, consistent with the saturation observed in Figure 8. Height predictions (row 3) better preserve spatial structure but underestimate peak values in tall forest areas (yellow-green regions in ground truth appear darker in predictions).

## Multi-Task Learning Model

### Experimental Setup

The multi-task U-Net was trained to jointly predict all three forest attributes using the same training data and hyperparameters as the baseline models. The shared encoder architecture (base channels = 128, dropout = 0.2) feeds three task-specific decoder branches for segmentation, height, and biomass prediction.

**Table 4: Multi-Task Model Training Configuration**

| Parameter | Value |
|-----------|-------|
| Base channels | 128 |
| Dropout (segmentation) | 0.4 |
| Dropout (regression) | 0.2 |
| Batch size | 8 |
| Maximum learning rate | 6×10⁻⁴ |
| Minimum learning rate | 3×10⁻⁵ |
| Weight decay | 1×10⁻⁴ |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 10 epochs |
| Maximum epochs | 100 |
| Loss weighting | Uncertainty-based |
| Allometric constraint weight | 1×10⁻⁴ |
| Allometric parameters | α = 0.0673, β = 2.5 |

The model employed uncertainty-weighted loss to automatically balance task contributions during training. Three learnable uncertainty parameters (log σ²) were initialized to zero and optimized jointly with the model weights. To prevent numerical instability from varying loss magnitudes, each task loss was normalized by its batch-wise mean before applying uncertainty weighting, with the global scale restored after weighting.

The allometric constraint between height and biomass was incorporated with weight λ_allom = 1×10⁻⁴. The allometric parameters (α = 0.0673, β = 2.5) were derived from published equations for temperate mixed forests, representing average scaling relationships between height and biomass. This constraint provides weak supervision by penalizing predictions that violate known ecological relationships.

Training curves revealed distinct convergence patterns for the three tasks. The segmentation loss decreased rapidly in the first 10 epochs before plateauing, while regression losses showed more gradual improvement. The uncertainty parameters evolved during training, with the segmentation uncertainty (log σ²_seg = 0.291) settling lower than height (0.536) and biomass (0.731) uncertainties, indicating that the model found segmentation easier to optimize relative to its loss magnitude.

Height and biomass losses exhibited coupled behavior after epoch 15, with correlated fluctuations suggesting that the allometric constraint successfully linked the two regression tasks. The validation metrics for all three tasks improved steadily until epoch 35, after which segmentation and height metrics stabilized while biomass R² continued to increase slightly, reaching 0.039 at convergence.

The learned task weights (derived from uncertainty parameters) converged to approximately 1.0 for segmentation, 0.4 for height, and 0.3 for biomass, reflecting the relative difficulty and loss scale of each task. These weights differ substantially from uniform weighting, demonstrating the value of automatic task balancing.

### Results

**Table 5: Test Set Performance of Multi-Task Model**

| Task | Metric | Value | Baseline | Change |
|------|--------|-------|----------|--------|
| **Segmentation** | Pixel Accuracy | 0.8528 | 0.8572 | -0.51% |
| | Mean IoU | 0.2821 | 0.2882 | -2.12% |
| | Mean Dice | 0.3657 | 0.3657 | 0.00% |
| **Height Regression** | RMSE (m) | 5.209 | 5.243 | -0.65% |
| | R² | 0.2645 | 0.255 | +3.73% |
| **Biomass Regression** | RMSE (t/ha) | 38.712 | 39.003 | -0.75% |
| | R² | 0.0390 | 0.024 | +62.5% |

The multi-task model achieved comparable performance to the baselines for segmentation and height, with slight improvements in height R² (+3.7%) and biomass R² (+62.5% relative improvement, though still low in absolute terms). The segmentation performance remained essentially unchanged (Dice = 0.366), suggesting that the shared encoder does not substantially help or hinder genus classification.

The modest improvement in height prediction (R² from 0.255 to 0.265) indicates that sharing representations with segmentation and biomass tasks provides weak regularization benefits. The larger relative improvement in biomass R² (0.024 to 0.039), while still representing poor absolute performance, suggests that the allometric constraint and shared features with height estimation help the model learn more structured biomass predictions.

<img src="../@plots/training-results/mtl/height-biomass-scatter-plot.png" width="100%" />

**Figure 10: Height and biomass prediction scatter plots for the multi-task model.** The height predictions (left) show similar patterns to the baseline model, with a dense cluster around 20-30m and systematic bias. The biomass predictions (right) exhibit the same saturation behavior as the baseline, with predictions confined to a narrow range despite wide variation in true values. The allometric constraint does not overcome the fundamental limitation of C-band SAR for high biomass estimation, though the slightly improved R² suggests more consistent predictions within the observable range.

<img src="../@plots/training-results/mtl/mtl-all-tasks-samples.png" width="100%" />

**Figure 11: Example predictions from the multi-task model on three test samples.** Comparing with Figure 9, the multi-task predictions show similar spatial patterns but with subtle differences. Segmentation predictions (row 1) maintain comparable quality to the baseline. Height predictions (row 2) appear slightly smoother, potentially due to regularization from the shared encoder. Biomass predictions (row 3) show marginally improved spatial coherence, particularly in the transition zones between high and low biomass areas, consistent with the allometric constraint encouraging physically plausible height-biomass combinations.

### Multi-Task Learning Analysis

To understand how the multi-task model learns shared and task-specific representations, we analyzed gradient alignment, feature similarity, and layer-wise representations during training.

#### Gradient Alignment in the Shared Encoder

<img src="../@plots/training-results/mtl/gradient-comparison.png" width="80%" />

**Figure 12: Gradient cosine similarity between task pairs in the shared encoder across training epochs.** The plot shows the cosine similarity between task-specific gradients computed on the shared encoder parameters. Positive values indicate aligned gradients (tasks agree on parameter updates), while negative values indicate conflicting gradients (tasks push parameters in opposite directions).

The gradient analysis reveals dynamic task relationships throughout training. In the first 15 epochs, all three task pairs exhibit highly variable gradient alignment, with frequent sign changes indicating that tasks initially compete for shared encoder capacity. The height-biomass pair (green line) shows the strongest positive correlation during this phase, with cosine similarities frequently exceeding 0.6, suggesting that these two regression tasks naturally share low-level feature requirements.

After epoch 15, gradient patterns stabilize. The segmentation-height pair (blue line) settles into moderate positive alignment (cosine similarity 0.4-0.7), indicating that genus classification and height estimation benefit from similar mid-level features. This makes ecological sense: both tasks require the model to distinguish forest structure, with genus affecting canopy architecture and height representing vertical structure.

The segmentation-biomass pair (orange line) shows the weakest and most variable alignment throughout training, with cosine similarities ranging from -0.3 to 0.6. This suggests that genus classification and biomass estimation require partially conflicting features. Biomass depends on both canopy structure and density, which may not align with genus-specific backscatter patterns. The frequent negative gradients indicate that improving biomass prediction sometimes requires encoder adjustments that harm segmentation performance.

The height-biomass pair maintains strong positive alignment (0.4-0.9) after the initial training phase, particularly from epochs 20-40. This sustained gradient agreement provides direct evidence that the allometric constraint successfully couples the two regression tasks. The high cosine similarity indicates that the model learns encoder features that simultaneously benefit both height and biomass prediction, likely capturing SAR patterns related to overall forest structure rather than task-specific details.

Notably, all task pairs show increased gradient alignment in the final training epochs (35-45), suggesting that the model converges to a shared representation that reasonably satisfies all three tasks. The reduced gradient conflict in this phase indicates that the uncertainty weighting successfully balanced task contributions, preventing any single task from dominating the shared encoder.

#### Feature Similarity Across Decoder Layers

<img src="../@plots/training-results/mtl/cka.png" width="100%" />

**Figure 13: Centered Kernel Alignment (CKA) similarity between task-specific decoder representations at different layers.** Each heatmap shows CKA similarity between decoder layers of two tasks, with rows representing the second decoder head and columns representing the first decoder head. Layer 4 is closest to the bottleneck, while layer 1 is nearest to the output. Higher CKA values (red) indicate more similar representations.

CKA analysis quantifies how similarly the task-specific decoders transform shared encoder features. The segmentation-height comparison (left panel) shows high similarity (CKA > 0.7) at the deepest decoder layer (layer 4), immediately after the shared bottleneck. This indicates that both tasks initially process the bottleneck features in similar ways, extracting common structural information about the forest canopy.

As we move toward shallower layers (layers 3, 2, 1), the CKA similarity decreases progressively, reaching 0.51-0.66 at layer 3 and 0.51-0.54 at layers 2 and 1. This gradient of decreasing similarity demonstrates that the decoders gradually specialize for their respective tasks. The segmentation decoder learns to emphasize genus-discriminative features (texture patterns, backscatter intensity variations), while the height decoder focuses on features related to vertical structure (volume scattering, canopy roughness).

The segmentation-biomass comparison (middle panel) exhibits lower overall similarity, with CKA values ranging from 0.25 to 0.51. The deepest layer (layer 4) shows moderate similarity (CKA ≈ 0.5), substantially lower than the segmentation-height pair. This confirms that biomass estimation requires fundamentally different feature processing than genus classification, even when starting from the same bottleneck representation.

The layer-wise CKA pattern for segmentation-biomass is relatively flat (0.25-0.51 across all layers), suggesting that these tasks diverge immediately after the bottleneck rather than gradually specializing. This supports the gradient analysis showing weak and variable alignment between segmentation and biomass tasks. The consistently low CKA values indicate that the segmentation decoder learns spatial patterns related to genus boundaries, while the biomass decoder must extract information about canopy density and structure that is largely orthogonal to genus classification.

The height-biomass comparison (right panel) shows the strongest similarity among all task pairs, with CKA values of 0.77-0.81 at layer 4 and remaining high (0.64-0.81) through layers 3, 2, and 1. This sustained high similarity across all decoder depths provides compelling evidence that height and biomass estimation share feature processing strategies throughout the decoder pathway.

Critically, the height-biomass CKA remains elevated even at the shallowest layers (layer 1: CKA = 0.70-0.81), where task-specific refinement should be strongest. This indicates that the allometric constraint successfully enforces consistent feature learning between the two regression tasks. The decoders learn to extract complementary information about forest structure (height focuses on vertical extent, biomass on density) while maintaining aligned representations that respect the ecological relationship between these variables.

The CKA analysis reveals a clear hierarchy of task relatedness: height-biomass (most similar) > segmentation-height (moderately similar) > segmentation-biomass (least similar). This hierarchy aligns with ecological expectations—height and biomass are physically coupled through allometry, height and genus both relate to canopy structure, while genus and biomass have weaker direct relationships.

#### Layer-wise Representation Analysis

<img src="../@plots/training-results/mtl/representations-across-layers.png" width="100%" />

**Figure 14: Visualization of intermediate representations across encoder and decoder layers for a single test sample.** The figure shows feature maps from the shared encoder (top two rows) and task-specific decoders (bottom three rows) at different network depths. Each column represents a different layer, progressing from input (left) to output (right).

The representation analysis provides direct visualization of how the network transforms SAR backscatter into task-specific predictions. The input VV and VH channels (leftmost panels) show the characteristic speckle pattern of SAR imagery, with brighter values indicating stronger backscatter. The spatial structure visible in the inputs—linear features corresponding to forest edges and textural variations within forest stands—provides the raw information for all downstream tasks.

The first encoder layer E1 (128×64×64) produces feature maps that emphasize edges and local texture patterns. The visualization shows enhanced contrast at forest boundaries and within-stand variations, indicating that the initial convolutions extract basic spatial structure from the SAR speckle. These low-level features are shared across all tasks, as evidenced by the high CKA similarity at deep decoder layers.

As we progress through the encoder (E2: 256×32×32, E3: 512×16×16), the feature maps become increasingly abstract. E2 shows larger-scale patterns corresponding to forest stand structure, while E3 captures even coarser spatial organization. The bottleneck (1024×8×8) produces highly compressed representations where individual spatial locations integrate information from large receptive fields (approximately 32×32 pixels or 800×800 meters).

The bottleneck visualization reveals structured patterns rather than random noise, indicating that the shared encoder successfully learns meaningful representations despite the competing task objectives. The visible spatial organization in the bottleneck features suggests that the model identifies coherent forest structures (stands, clearings, boundaries) that are relevant to multiple tasks.

The segmentation decoder (s2, s3, s4) progressively refines these representations into genus-specific patterns. At s4 (128×64×64, near the bottleneck), the features show coarse spatial structure similar to the encoder. By s2 (256×32×32), distinct spatial regions corresponding to different genera begin to emerge, visible as areas with different activation patterns. The final segmentation output shows sharp genus boundaries, demonstrating that the decoder successfully recovers fine spatial detail through the skip connections.

The height decoder (h2, h3, h4) shows smoother spatial patterns than segmentation, consistent with the continuous nature of the height prediction task. The h4 features (128×64×64) exhibit gradual spatial variations rather than sharp boundaries, suggesting that the decoder learns to extract information about canopy height from volume scattering and backscatter intensity patterns. The intermediate layers (h3, h2) show progressive refinement of these smooth patterns, with the final height prediction displaying realistic spatial gradients from low to high canopy areas.

The biomass decoder (b2, b3, b4) produces the most distinct feature patterns among the three tasks. The b4 features show a different spatial organization than either segmentation or height, with emphasis on regions of high backscatter intensity. This aligns with the expectation that biomass estimation relies on canopy density information encoded in backscatter strength. However, the relatively uniform activation patterns in the biomass decoder intermediate layers (b3, b2) reflect the saturation problem—the decoder struggles to differentiate high biomass areas because the C-band SAR signal provides limited information beyond a threshold.

Comparing the final predictions with ground truth reveals the strengths and limitations of each task. The segmentation prediction captures the major genus boundaries but misses some fine-scale transitions. The height prediction preserves the overall spatial pattern but underestimates peak values, visible as reduced intensity in tall forest areas. The biomass prediction shows the most severe degradation, with compressed dynamic range and loss of fine spatial detail, directly reflecting the low R² performance.

Critically, the representation analysis demonstrates that the shared encoder learns features that are genuinely useful for multiple tasks. The structured patterns visible at all encoder depths, combined with the task-specific refinement in the decoders, confirm that multi-task learning successfully extracts common SAR backscatter patterns while allowing task-specific specialization. The smooth transition from shared to specialized representations across the network depth validates the architectural design of shared encoder with task-specific decoders.

The visualization also reveals where feature sharing occurs. The encoder layers (E1-E3) show similar activation patterns regardless of which task's decoder we examine, confirming that these representations are truly shared. In contrast, the decoder layers (s2-s4, h2-h4, b2-b4) show increasingly divergent patterns, with each task's decoder transforming the shared bottleneck features according to its specific requirements.

This analysis provides concrete evidence that the multi-task model learns hierarchical representations: low-level features (edges, textures) are shared across all tasks, mid-level features (forest structure, stand patterns) are partially shared with task-specific emphasis, and high-level features (genus boundaries, height gradients, biomass patterns) are task-specific. The successful learning of this hierarchy, despite the competing task objectives revealed in the gradient analysis, demonstrates that the uncertainty weighting and allometric constraint effectively balance the multi-task optimization.

### Summary

The experimental results demonstrate that multi-task learning provides modest benefits for forest attribute estimation from Sentinel-1 SAR data. The multi-task model achieves comparable performance to single-task baselines while using a single shared encoder, offering computational efficiency without sacrificing accuracy. The uncertainty-weighted loss successfully balances task contributions, as evidenced by the stable convergence and reasonable task weight distribution.

The gradient analysis reveals that height and biomass tasks naturally align in their feature requirements, while segmentation shows weaker alignment with both regression tasks. The allometric constraint successfully couples height and biomass prediction, maintaining high gradient similarity and CKA scores throughout training. The CKA analysis demonstrates clear task specialization in the decoder layers, with height-biomass showing the strongest feature similarity and segmentation-biomass the weakest.

The representation analysis confirms that the shared encoder learns meaningful features for all tasks, with progressive specialization in the task-specific decoders. The visible spatial structure in intermediate representations indicates that the model successfully extracts forest structural information from SAR backscatter, though fundamental limitations of C-band SAR (height sensitivity, biomass saturation) constrain absolute performance.

While the multi-task model does not substantially outperform the baselines, it demonstrates that joint training is feasible and provides regularization benefits, particularly for the challenging biomass estimation task. The analysis tools (gradient alignment, CKA, representation visualization) provide clear evidence that the model learns shared features in the encoder and task-specific features in the decoders, validating the multi-task learning approach for this application.
