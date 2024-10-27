### Project: ReConPatch - Contrastive Patch Representation Learning for Industrial Anomaly Detection

#### Overview
The **ReConPatch** project introduces a novel approach for industrial anomaly detection by leveraging **contrastive patch representation learning**. Industrial anomaly detection is crucial in maintaining product quality across manufacturing sectors. Traditional machine learning techniques often struggle to detect rare and complex defects due to their unpredictable nature. ReConPatch addresses these challenges by adapting visual representations from pre-trained models, transforming them into highly separable feature spaces optimized for detecting subtle defects without requiring extensive handcrafted augmentations.

#### Team Composition
- **Team Leader**: Ishan Kanodia (22AE10014)
- **Team Members**: Divyanshu Bhatt (22GG10017), Adeeba Alam Ansari (22IM30002), Satya Ravi Sujit (21IM3FP10), Anumothu Ankit (21MF10008), G Sreekumar (20GG20021)

---

#### Project Background and Motivation
In industrial manufacturing, **defects** such as **misalignments, incorrect parts, and surface damage** can severely impact product quality. Given that defects are rare and exhibit diverse characteristics, traditional anomaly detection models (e.g., autoencoders and GANs) often fail to generalize effectively to unseen anomalies. Standard methods for anomaly detection rely on one-class classification, which distinguishes anomalies by their distance from the nominal distribution. However, such models struggle with industrial data due to shifts between natural and industrial image domains.

ReConPatch bridges this gap by using **pre-trained image representations** (from natural datasets) and adapting them for industrial settings through **contrastive learning** on patches within the images. This approach enables the detection of fine-grained industrial anomalies by creating a **target-oriented feature space** where normal and defective instances are highly distinguishable.

---

#### Methodology

**1. Training Phase**
   - The model extracts **patch-level features** from images using a CNN pre-trained on a large, diverse dataset.
   - These features are processed to create **pairwise and contextual similarity measures** as pseudo-labels.
   - Using these pseudo-labels, ReConPatch employs **relaxed contrastive loss**, training the feature space to bring similar patches close together and push dissimilar patches apart.
   - The **pairwise similarity** score relies on a Gaussian kernel-based similarity between patch embeddings, while **contextual similarity** evaluates the neighborhood structure around each feature, ensuring only relevant similarities are emphasized.
   - To stabilize training, **exponential moving average (EMA)** updates the similarity calculation network progressively.

**2. Inference Phase**
   - In inference, patch features of a test image are computed and matched to representative features (stored in a **memory bank**) to determine anomaly scores.
   - An **anomaly score** is calculated by comparing the patch to its nearest nominal representative, facilitating both **pixel-level and image-level anomaly detection**.
   - An **ensemble of models** (WideResNet-101, ResNext-101, and DenseNet-201) further enhances ReConPatch’s accuracy by capturing diverse feature representations.

#### Technical Details and Hyperparameter Tuning
   - ReConPatch was developed in **Python 3.7** using **PyTorch**.
   - Feature extraction leverages **WideResNet-50** and **WideResNet-101** backbones.
   - Optimal patch size and feature hierarchy were identified using ablation studies, with a **patch size of 3** and hierarchical levels set to 1, 2, and 3 for most datasets.
   - Key hyperparameters such as **k (nearest neighbors)**, **repelling margin (m)**, and the **linear combination ratio (α)** for similarity calculation were tuned extensively:
     - **k**: Optimal value was found at 5 for balancing anomaly separation.
     - **m**: Set to 1 for effective contrastive training.
     - **α**: Balanced at 0.5 for equal weight between pairwise and contextual similarities, maximizing AUROC in detection.

#### Experimental Setup and Results

The project benchmarks ReConPatch on two widely-used datasets: **MVTec AD** and **BTAD**.

**1. MVTec AD Dataset**
   - **Dataset Characteristics**: Consists of 15 classes across various industrial objects, with normal and defective images totaling 5,354 images.
   - **Metrics**: AUROC (Area Under Receiver Operating Characteristic curve) is used for both **image-level anomaly detection** and **pixel-level segmentation**.
   - **Results**: 
     - Single model achieved **99.56% image-level AUROC** and **98.07% pixel-level AUROC**.
     - Ensemble model further enhanced performance, reaching **99.72% detection AUROC** and **98.67% segmentation AUROC**.
   - **Ablation Study**: Tested different patch sizes and hierarchy levels to optimize performance. For example, including hierarchy levels 1, 2, and 3 improved segmentation accuracy to 98.18%.

**2. BTAD Dataset**
   - **Dataset Characteristics**: Includes images of three distinct industrial components with both normal and defective variations.
   - **Results**: ReConPatch achieved **95.8% image-level AUROC** and **97.5% segmentation AUROC**, surpassing contemporary methods like PatchCore.

#### Visualizations and Qualitative Analysis
- **Feature Space Visualization**: UMAP (Uniform Manifold Approximation and Projection) was used to project high-dimensional patch features into a 2D space, showing clusters of nominal and defective patches distinctly separated.
- **Anomaly Score Distribution**: ReConPatch produced a clear distribution separation between normal and abnormal patches, enhancing discrimination capabilities.
- **Anomaly Score Maps**: Heatmaps display anomaly scores, with detected anomalies highlighted for various object classes in the MVTec AD dataset. Ground truth anomalies align closely with ReConPatch predictions, validating the model’s spatial accuracy.

#### Future Directions
1. **Optimization of Hyperparameters**: Further refinement of k, m, and α to tailor ReConPatch to more diverse industrial datasets.
2. **Enhanced Visualizations**: Incorporating advanced anomaly visualization techniques, including AUROC-based performance metrics and more complex feature space analyses.
3. **Extended Applications**: Deploying ReConPatch in real-world industrial inspection systems to test robustness under different environmental conditions like variable lighting and complex object orientations.

#### Conclusion
ReConPatch establishes a robust framework for **unsupervised anomaly detection** in industrial contexts. By adapting contrastive patch representation learning, it effectively separates defective patches from nominal ones, even under challenging conditions without data augmentation. This breakthrough provides a practical, deployable solution to enhance quality control processes in manufacturing.
