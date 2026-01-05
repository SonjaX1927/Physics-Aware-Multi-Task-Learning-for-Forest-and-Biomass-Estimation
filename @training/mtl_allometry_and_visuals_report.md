# Refining Multi-Task Learning for Tree Allometry and Visual Diagnostics

## Summary
- **Goal.** Develop a multi-task U-Net that jointly predicts tree genus, height, and biomass while enforcing a biologically meaningful allometry relationship between height and biomass.
- **Concern 1.** The original allometry loss was species-agnostic, which ignored known variation in height–biomass scaling across genera and could bias training.
- **Concern 2.** The different task losses operated on very different numeric scales, which made uncertainty-based loss weighting unstable and hard to interpret.
- **Concern 3.** It was unclear how strongly the three task heads shared or conflicted in their gradients, and whether the shared encoder was being pulled in incompatible directions.
- **Concern 4.** The first implementation of CKA similarity focused on within-branch structure rather than similarity between task heads, which did not answer the main scientific question.
- **Concern 5.** The original feature visualizations were fragmented and made it hard to see how encoder and decoder representations evolved jointly across tasks.
- **Solution overview.** I introduced a species-aware allometry loss, normalized losses for uncertainty weighting, added gradient cosine and cross-head CKA analyses, redesigned the feature visualization layout, and cleaned up the evaluation API to make the system more consistent and interpretable.

---

## 1. Background and Main Concerns

The project builds a multi-task learning (MTL) model that shares an encoder between three heads: a segmentation head for genus prediction, a height regression head, and a biomass regression head. The key scientific motivation is to use the strong physical link between tree height and biomass to regularize the model through an allometry constraint.

Initially, the allometry regularization was defined as a simple global relationship between predicted height and biomass. This raised several concerns:

- **Species variation.** Different genera are known to follow different height–biomass relationships, so a single global allometry curve risks over-penalizing some genera while under-penalizing others.
- **Loss scale mismatch.** The segmentation, height, biomass, and allometry losses had different scales, which complicated the use of uncertainty-based weighting and could make some losses dominate the optimization.
- **Gradient conflict.** With three tasks, there was a risk that their gradients on the shared encoder would conflict, but this was not monitored explicitly.
- **Representation sharing.** It was unclear to what extent the three decoders were learning similar or different representations at each resolution level.
- **Visual diagnosis.** Existing visualizations of feature maps and predictions were not aligned to the questions about information flow through the encoder and decoders.

The work in this session addresses these concerns by making the allometry loss species-aware, stabilizing the weighting scheme, and adding tools to inspect gradients and features.

---

## 2. Multi-Task U-Net and Allometry Regularization

The base model is a U-Net with a shared encoder and three separate decoders. Each decoder operates at four resolution levels, with a bottleneck at the coarsest scale. The tasks are:

- **Genus segmentation.** Predict a genus class for each pixel.
- **Height regression.** Predict per-pixel height.
- **Biomass regression.** Predict per-pixel biomass.

The original allometry regularizer enforced a global log–log relationship between height and biomass of the form

$$
B \approx e^{\alpha} H^{\beta}
$$

which is equivalently written in log–log space as

$$
\log B \approx \alpha + \beta \log H.
$$

The loss penalizes the squared difference between the predicted \(\log B\) and the allometry-based \(\log B\), applied uniformly across all pixels and genera.

This formulation captured the general shape of the relationship but did not reflect genus-specific differences, and it treated all valid pixels equally, regardless of the model's confidence about the genus.

---

## 3. Species-Aware Allometry Loss

To respect genus-specific differences while still working with image-level predictions, I moved from a global allometry constraint to a **species-aware** one.

### 3.1. Conceptual Formulation

Instead of assuming a single pair of parameters \(\alpha, \beta\), I define vectors \(\boldsymbol{\alpha}\) and \(\boldsymbol{\beta}\), one pair per genus. For each pixel, the segmentation head outputs a probability distribution over genera. The expected allometric log-biomass for that pixel is then defined as a **mixture over genera**, using the predicted genus probabilities as weights. The allometry loss is a squared error between the predicted \(\log B\) and this expected allometric \(\log B\), restricted to pixels where height is above a configurable minimum.

I formalize the expected allometric log-biomass at pixel location \(x\) as

$$
\log B_{\text{allo}}(x) = \sum_{c=1}^{C} p_c(x)\,\bigl(\alpha_c + \beta_c \log H(x)\bigr),
$$

where \(p_c(x)\) is the predicted probability of genus \(c\) at \(x\), and \(H(x)\) is the predicted height.

The corresponding allometry loss over pixels is

$$
\mathcal{L}_{\text{allom}} = \mathbb{E}_{x}\Bigl[ \mathbf{1}_{\{H(x) > H_{\min}\}}\,\bigl(\log B_{\text{pred}}(x) - \log B_{\text{allo}}(x)\bigr)^2 \Bigr],
$$

where \(H_{\min}\) is a minimum height threshold and \(B_{\text{pred}}(x)\) is the predicted biomass.

This design has several advantages:

- **Soft species assignment.** It does not require hard genus labels at prediction time; it uses the model's own segmentation probabilities.
- **Genus sensitivity.** If the model shifts probability mass between genera with different allometry parameters, the expected allometric prediction changes accordingly.
- **Consistent integration.** The loss uses the same predictions that drive the segmentation and regression heads, so it is naturally integrated into the MTL framework.

### 3.2. Practical Parameterization

For now, the implementation constructs per-genus \(\boldsymbol{\alpha}\) and \(\boldsymbol{\beta}\) vectors by broadcasting the scalar allometry parameters from the configuration, so the code runs without needing external species-specific values.

I also discussed how to later replace these default vectors with genus-specific values estimated from data (see Section 7), without changing the training logic.

### 3.3. Integration into Training and Evaluation

The species-aware allometry loss is fully wired into both the training and evaluation loops:

- **Training.** The `train_one_epoch_mtl` function now computes the species-aware allometry loss using segmentation logits, height predictions, biomass predictions, and the per-genus parameter vectors, and adds it to the total loss with a configurable weight.
- **Evaluation.** The `evaluate` function uses the same species-aware loss for logging, so that training and evaluation metrics are consistent.
- **API consistency.** Duplicate evaluation definitions were removed, and the evaluation signature was cleaned up to accept and forward allometry parameters in a single, consistent way.

---

## 4. Loss Weighting and Uncertainty Normalization

The model supports several loss weighting strategies, including fixed weights, uncertainty-based weighting (à la Kendall), and GradNorm-style adaptive weighting. A practical difficulty was that the raw segmentation, height, biomass, and allometry losses lived on different numeric scales.

To make the uncertainty-based weighting behave more predictably, I introduced **pre-normalization** of the task losses:

- **Pre-normalized losses.** Before applying the uncertainty weighting formula, each task loss is divided by a configurable scale factor. These factors are chosen so that the normalized losses are roughly on the same order of magnitude.
- **Logging of raw values.** For interpretability, the original, unnormalized losses are still logged. Only the internal computation of the uncertainty-weighted combined loss uses the normalized values.

Concretely, for each task \(i\) with raw loss \(L_i\) and scale factor \(\kappa_i\), I define a normalized loss

$$
\tilde{L}_i = \frac{L_i}{\kappa_i}.
$$

Using Kendall-style uncertainty weighting, with learnable log-variances \(s_i = \log \sigma_i^2\), the combined task loss is

$$
\mathcal{L}_{\text{unc}} = \sum_{i} \left( \frac{1}{2\sigma_i^{2}}\,\tilde{L}_i + \frac{1}{2}\log \sigma_i^{2} \right).
$$

In all cases, the **total multi-task loss** combines the task part and the allometry regularizer as

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{allom}}\,\mathcal{L}_{\text{allom}},
$$

where \(\mathcal{L}_{\text{task}}\) is either a fixed-weight sum of task losses,

$$
\mathcal{L}_{\text{task}} = w_{\text{seg}} L_{\text{seg}} + w_{\text{h}} L_{\text{h}} + w_{\text{b}} L_{\text{b}},
$$

or the uncertainty-weighted objective \(\mathcal{L}_{\text{unc}}\), and \(\lambda_{\text{allom}}\) is the configurable weight on the allometry loss.

This strategy keeps the probabilistic interpretation of the uncertainty weighting while avoiding degenerate behavior caused by large scale differences between tasks.

---

## 5. Gradient Interactions and Similarity Analyses

Understanding how the three tasks interact in the shared encoder is important both for optimization stability and for scientific interpretability. Two main tools were added or refined for this purpose: gradient cosine similarity and CKA similarity.

### 5.1. Gradient Cosine Similarity

I measure the cosine similarity between gradients of the three task losses with respect to shared encoder parameters. This is done on at least one representative batch during each epoch.

- **Positive cosine.** Indicates that two tasks are pushing the encoder parameters in similar directions, which suggests synergistic learning.
- **Negative cosine.** Indicates gradient conflict, where improving one task could temporarily harm another.

Formally, for two gradient vectors \(g_i\) and \(g_j\), I compute

$$
\cos(g_i, g_j) = \frac{g_i^\top g_j}{\lVert g_i \rVert_2\,\lVert g_j \rVert_2}.
$$

These metrics are recorded in the training history and can be plotted over epochs, providing a direct view of how the multi-task coupling evolves during training.

### 5.2. Cross-Head CKA Similarity

The initial CKA implementation focused on similarity within each decoder branch across layers, which did not directly answer how similar different tasks were to each other at a given depth.

I redesigned the CKA analysis to compute **pairwise CKA between decoder heads** at each resolution level:

- **Head pairs.** Segmentation vs height, segmentation vs biomass, and height vs biomass.
- **Layers.** For each pair, I compare feature maps at the corresponding decoder layers (e.g., the four upsampling levels).

The result is a set of 4×4 CKA matrices for each head pair, visualized in a compact figure. This directly shows where the different tasks share internal representations and where they diverge.

For two centered Gram matrices \(K_c\) and \(L_c\) built from features of two heads, I use the standard CKA definition

$$
\operatorname{CKA}(K, L) = \frac{\langle K_c, L_c \rangle_F}{\lVert K_c \rVert_F\,\lVert L_c \rVert_F},
$$

where \(\langle \cdot, \cdot \rangle_F\) is the Frobenius inner product.

---

## 6. Feature Visualization and Decoder Artifacts

To better understand what the encoder and decoders learn, I redesigned the feature visualization into a structured **4×6 grid** layout. This layout organizes information as follows:

- **Row 1.** Input channels (VV, VH) and early encoder features (E1, E2, E3, bottleneck).
- **Row 2.** Genus-related information: genus index, segmentation prediction, and segmentation decoder features.
- **Row 3.** Height ground truth, height prediction, and height decoder features.
- **Row 4.** Biomass ground truth, biomass prediction, and biomass decoder features.

Each feature map is shown with its channel–height–width shape in the title, making it easier to link the visualization back to the model architecture.

I also discussed the presence of **checkerboard artifacts** in some decoder feature maps, which are a known side effect of transpose convolution–based upsampling. While the current architecture keeps these layers, the visualization now makes such artifacts visible and allows you to judge whether they are acceptable or require architectural changes (e.g., switching to resize–convolution).

---

## 7. Fitting Genus-Specific Allometry Parameters from Data

The species-aware allometry loss is designed to accept per-genus \(\boldsymbol{\alpha}\) and \(\boldsymbol{\beta}\) parameters, but the current implementation uses default values broadcast from scalars. To make the loss scientifically grounded, I outlined a practical procedure to estimate these parameters from your own dataset.

The general strategy is:

- **Data preparation.** For each pixel (or spatial unit) with valid height and biomass measurements and a known genus label, compute \(\log H\) and \(\log B\) after applying a minimum height threshold.
- **Per-genus regression.** For each genus, fit a simple linear regression of \(\log B\) on \(\log H\), yielding an estimated \(\alpha_g\) and \(\beta_g\).

For each genus \(g\), I write the regression model as

$$
\log B = \alpha_g + \beta_g \log H + \varepsilon,
$$

where \(\varepsilon\) is a residual term.
- **Regularization.** For genera with few samples, consider sharing information across genera (e.g., partial pooling, or shrinking towards a global mean) to avoid overfitting.
- **Integration.** Store the resulting vectors of \(\alpha_g\) and \(\beta_g\) and pass them into the training loop, replacing the broadcast default values.

This approach ensures that the allometry regularizer reflects empirical relationships in your data while still being flexible enough to incorporate literature-based values where available.

---

## 8. Remaining Open Questions and Future Work

Several directions remain open for further refinement:

- **Genus parameter robustness.** Once per-genus \(\boldsymbol{\alpha}\) and \(\boldsymbol{\beta}\) are estimated, it will be useful to study how sensitive training and evaluation metrics are to these values.
- **Advanced gradient surgery.** If gradient cosine similarity consistently shows strong conflicts, methods like PCGrad or other gradient surgery techniques could be added to further stabilize multi-task optimization.
- **Alternative allometry forms.** The current loss assumes a log–log linear relationship; if empirical data suggest systematic deviations, more flexible forms could be explored.
- **Architectural changes.** If checkerboard artifacts or CKA analyses reveal strong bottlenecks in representation sharing, the decoder architectures could be adjusted to encourage or discourage sharing at specific depths.

Overall, the modifications made so far move the model towards a more faithful integration of species-dependent allometry, more stable loss weighting, and richer diagnostics for both optimization and representation learning.
