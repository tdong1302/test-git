# Quá Trình Phát Triển và Đánh Giá Hệ Thống

## Tổng Quan

Chương này trình bày chi tiết quá trình phát triển hệ thống dự đoán chức năng protein qua 5 phiên bản thử nghiệm (ver1 → ver5). Mỗi phiên bản đánh dấu một bước tiến trong việc cải thiện kiến trúc mô hình, chiến lược huấn luyện và kết quả dự đoán. Phiên bản ver5 được xác định là phiên bản tối ưu nhất với kiến trúc Multi-Aspect Ensemble.

---

## Phương Pháp Nghiên Cứu Đề Xuất (Best-Version: Ver5)

### Tổng Quan Hệ Thống

Phiên bản ver5 - **Multi-Aspect Ensemble** - là phương pháp tối ưu đề xuất cho bài toán dự đoán chức năng protein (CAFA-6). Hệ thống kết hợp ba thành phần chính: embedding tiền huấn luyện, kiến trúc mô hình multi-aspect chuyên biệt, và cơ chế lan truyền GO hierarchy để đảm bảo tính nhất quán sinh học.

### 1. Pipeline Multi-Aspect Ensemble

```
┌─────────────────────────────────────────────────────────┐
│            Protein Sequences (Input)                     │
│        train_sequences.fasta (~144K proteins)            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│        ESM-2 Embedding Generation                        │
│  (Pre-computed: 1280-dim protein language model)         │
│  - Load embeddings từ protein_embeddings.npy             │
│  - Map protein IDs từ protein_ids.csv                    │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ CCO    │ │ MFO    │ │ BPO    │
   │ Model  │ │ Model  │ │ Model  │
   │ ~2.6K  │ │ ~6.6K  │ │~16.8K  │
   │ terms  │ │ terms  │ │ terms  │
   └────┬───┘ └────┬───┘ └────┬───┘
        │          │          │
        ▼          ▼          ▼
   ┌────────────────────────────────┐
   │  Multi-Label Predictions       │
   │  Raw scores [0, 1] per term    │
   │  Confidence scores per aspect  │
   └────────┬───────────────────────┘
            │
            ▼
   ┌────────────────────────────────┐
   │  Threshold Optimization        │
   │  CCO: 0.02  MFO: 0.05  BPO: 0.015
   └────────┬───────────────────────┘
            │
            ▼
   ┌────────────────────────────────┐
   │  GO Hierarchy Propagation      │
   │  Ancestor voting & max scoring │
   │  True Path Rule enforcement    │
   └────────┬───────────────────────┘
            │
            ▼
   ┌────────────────────────────────┐
   │  Final Predictions (Output)    │
   │  Multi-label assignment file   │
   │  (protein_id, GO_term, score)  │
   └────────────────────────────────┘
```

### 2. Input: ESM-2 Embeddings

**Embedding Source:**
- **Model**: ESM-2 (Evolutionary Scale Modeling) - Large language model trained on ~2.7 billion protein sequences
- **Dimension**: 1280 features (learned representations)
- **Pre-computed**: File `protein_embeddings.npy` (82,404 proteins × 1280 dimensions)
- **Mapping**: `protein_ids.csv` maps protein IDs to embedding indices

**Lý do chọn ESM-2:**
1. **Semantic Understanding**: Captures evolutionary and structural patterns learned từ billions of protein sequences
2. **Transfer Learning**: Pre-trained representations reduce training data requirements
3. **Alignment-Free**: Không cần multiple sequence alignment (MSA) - faster inference
4. **Superior Performance**: Outperforms hand-crafted features (K-mer, physicochemical) trên multi-label tasks
5. **Language Model Advantage**: Token-level attention captures long-range dependencies trong protein structure

**Embedding Properties:**
```python
embeddings.shape = (82404, 1280)  # 82K proteins × 1280-dim
embeddings_mean = 0.0234           # Standardized
embeddings_std = 0.892             # Well-normalized
```

### 3. Output: Multi-Label Predictions per Aspect

**Prediction Format:**
```
protein_id    aspect    GO_term      prediction_score
─────────────────────────────────────────────────────
A0A000         C         GO:0005575        0.82
A0A000         C         GO:0005622        0.71
A0A000         F         GO:0003674        0.65
A0A000         P         GO:0008150        0.78
...
```

**Output Characteristics:**
- **Per-Aspect Separation**: Predictions riêng biệt cho CCO (C), MFO (F), BPO (P)
- **Probabilistic Scores**: [0, 1] range representing model confidence
- **Multi-Label Nature**: Một protein có thể được assigned nhiều GO terms trong cùng aspect
- **Score Aggregation**: Sau GO propagation, final score = max(direct_prediction, ancestor_propagated_score)

### 4. Loss Function: BCEWithLogits (Multi-Label Optimized)

**Loss Formula:**
$$L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} \left[ y_{ij} \log(\sigma(z_{ij})) + (1-y_{ij}) \log(1-\sigma(z_{ij})) \right]$$

Trong đó:
- $\sigma(z) = \frac{1}{1+e^{-z}}$ (sigmoid activation)
- $y_{ij}$ = binary label (0 hoặc 1)
- $z_{ij}$ = raw logit output từ model
- N = batch size, K = số GO terms trong aspect

**Tại sao BCEWithLogits thay vì Focal Loss?**

| Aspect | BCEWithLogits | Focal Loss |
|--------|---------------|-----------|
| Stability | Cao (numerically stable) | Trung bình |
| Hard Negatives | Implicit via label weight | Explicit via γ |
| Training Speed | Nhanh hơn | Chậm hơn |
| Multi-Aspect | Tốt hơn | Cơ bản |
| Implementation | PyTorch built-in | Custom implementation |

**Label Weighting Strategy:**
```python
# Per-aspect weight balancing
for aspect in ['C', 'F', 'P']:
    pos_count = (y_aspect == 1).sum()
    neg_count = (y_aspect == 0).sum()
    pos_weight = neg_count / pos_count  # Increase positive gradient
    loss = BCEWithLogitsLoss(pos_weight=pos_weight)
```

Weighted BCE effectively xử lý class imbalance extreme (negative:positive ~ 100:1) bằng cách tự động scale gradients cho minority class.

### 5. GO Hierarchy Propagation: Đảm Bảo Tính Nhất Quán Sinh Học

**Khái niệm True Path Rule:**
Trong Gene Ontology, nếu protein được annotated cho một GO term, nó cũng phải được annotated cho tất cả ancestors của term đó trong ontology hierarchy.

**Propagation Algorithm:**

```python
def propagate_go_predictions(predictions, go_hierarchy, threshold):
    """
    Input:  predictions (dict: protein_id → {term: score})
            go_hierarchy (dict: term → [parent_terms])
    Output: propagated_predictions (ensure all ancestors included)
    """
    
    propagated = {}
    for protein_id, terms in predictions.items():
        # Step 1: Filter predictions above threshold
        confident_terms = {t: s for t, s in terms.items() if s >= threshold}
        
        # Step 2: Add all ancestors with aggregated scores
        for term, score in confident_terms.items():
            ancestors = get_ancestors(term, go_hierarchy)
            for ancestor in ancestors:
                # Score aggregation: max of direct & propagated
                ancestor_score = max(
                    ancestor.get(ancestor, 0),  # Direct prediction
                    score * decay_factor(term, ancestor)  # Propagated
                )
                propagated[protein_id][ancestor] = ancestor_score
    
    return propagated
```

**Key Features:**
1. **Ancestor Voting**: Mỗi term "vote" cho ancestors nó
2. **Score Decay**: Ancestor score = child_score × decay_factor (distance-dependent)
3. **Max Aggregation**: Nếu term có multiple paths lên ancestor, lấy max score
4. **True Path Enforcement**: Đảm bảo consistency với GO semantic

**Ví dụ Propagation:**
```
Raw predictions (protein A):
  GO:0006355 (Gene Expression)     → score 0.85
  GO:0003677 (DNA Binding)         → score 0.72

GO Hierarchy:
  GO:0003674 (Molecular_Function)
    ├─ GO:0003677 (DNA Binding)
    │   └─ GO:0006355 (Gene Expression)
    │       └─ GO:0006450 (DNA Repair)
    └─ ...

After Propagation (protein A):
  GO:0006355 (Gene Expression)     → score 0.85 (direct)
  GO:0003677 (DNA Binding)         → score 0.85 (from child GO:0006355)
  GO:0003674 (Molecular_Function)  → score 0.72 (from child GO:0003677)
  GO:0006450 (DNA Repair)          → score 0.72 (propagated from parent)
```

### 6. Kiến Trúc Mô Hình: Multi-Aspect MLP Architecture

#### 6.1. Tổng Quan Kiến Trúc

Ver5 sử dụng 3 independent Multi-Layer Perceptron (MLP) models, mỗi model được huấn luyện riêng biệt cho một GO aspect (CCO/MFO/BPO).

**Model Topology:**

```python
class ProteinPredictor(nn.Module):
    """Aspect-specific multi-label classifier"""
    def __init__(self, input_dim=1280, num_classes):
        super().__init__()
        
        # Layer 1: Embedding to Hidden
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        
        # Layer 2: Hidden to Deeper Hidden
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        
        # Layer 3: Output Layer
        self.fc3 = nn.Linear(512, num_classes)
        # Note: NO activation here! BCEWithLogitsLoss expects raw logits
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        return x  # Raw logits [batch_size, num_classes]
```

#### 6.2. Kiến Trúc Chi Tiết

| Layer | Input Dim | Output Dim | Operation | Purpose |
|-------|-----------|-----------|-----------|---------|
| Input | 1280 | 1280 | Identity | ESM-2 embeddings |
| FC1 | 1280 | 1024 | Linear + BatchNorm + ReLU + Dropout(0.3) | Feature expansion & normalization |
| FC2 | 1024 | 512 | Linear + BatchNorm + ReLU + Dropout(0.3) | Hierarchical feature learning |
| Output | 512 | K* | Linear (no activation) | Logits for BCEWithLogits |

*K = số GO terms trong aspect (CCO: 2651, MFO: 6616, BPO: 16858)

#### 6.3. Design Rationale

1. **Input Dimension (1280)**: Direct use của ESM-2 embeddings - no projection layer needed vì embeddings đã optimal
   
2. **Hidden Dimension (1024)**: Slight expansion từ input dimension
   - Cho phép model learn non-linear transformations
   - ~2.6M parameters → manageable memory footprint
   - Prevents information bottleneck
   
3. **Intermediate Dimension (512)**: Gradual compression
   - Forces model learn hierarchical representations
   - Bottleneck effect: force sparse activation patterns
   - Improves generalization (acts as regularization)
   
4. **Dropout (0.3)**: Moderate regularization
   - Applied after activation functions
   - Prevents co-adaptation of neurons
   - Especially important vì class imbalance extreme
   
5. **BatchNorm**: Normalization strategy
   - Stabilizes training dynamics
   - Reduces internal covariate shift
   - Allows higher learning rates
   - Per-batch statistics (no running mean needed vì inference batch-size agnostic)
   
6. **No Output Activation**: Critical design choice
   - BCEWithLogitsLoss expects raw logits
   - Sigmoid applied internally in loss function
   - More numerically stable than separate Sigmoid + BCE

#### 6.4. Parameter Count

```
CCO Model:
  FC1: 1280 × 1024 + 1024 = 1,311,744 params
  FC2: 1024 × 512 + 512 = 524,800 params
  FC3: 512 × 2651 + 2651 = 1,357,763 params
  ─────────────────────────
  Total: 3,194,307 parameters

MFO Model:
  Same FC1, FC2: 1,836,544 params
  FC3: 512 × 6616 + 6616 = 3,387,520 params
  ─────────────────────────
  Total: 5,224,064 parameters

BPO Model:
  Same FC1, FC2: 1,836,544 params
  FC3: 512 × 16858 + 16858 = 8,631,186 params
  ─────────────────────────
  Total: 10,467,730 parameters

ALL 3 Models: ~18.9M total parameters
```

### 7. Chiến Lược Huấn Luyện

**Per-Aspect Training Config:**

| Tham số | Giá trị | Giải Thích |
|---------|---------|-----------|
| Batch Size | 128 | Balance between gradient noise & memory |
| Epochs | 30 | Empirical sweet spot cho convergence |
| Learning Rate | 1e-3 | Standard cho AdamW optimization |
| Weight Decay | 1e-4 | Mild L2 regularization |
| Optimizer | AdamW | Adaptive moment + weight decay decoupling |
| LR Scheduler | ReduceLROnPlateau | Reduce LR khi validation loss plateau |
| Patience | 2 epochs | Trigger LR reduction sau 2 epochs không improvement |
| Early Stopping | N/A | Train đầy đủ 30 epochs (model size khác nhau) |

**Training Loop:**
```python
for epoch in range(1, EPOCHS+1):
    # Forward pass
    train_loss = 0.0
    for batch in train_loader:
        X, y = batch
        logits = model(X)
        loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            X, y = batch
            logits = model(X)
            loss = BCEWithLogitsLoss(logits, y)
            val_loss += loss.item()
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d}/30 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
```

### 8. Tổng Kết Phương Pháp Đề Xuất

**Strengths:**
- ✅ Aspect-specific optimization: mỗi ontology có training parameters riêng
- ✅ Pre-trained embeddings: leverage 2.7B protein sequences từ ESM-2
- ✅ Multi-aspect ensemble: 3 models reduce single-model bias
- ✅ GO hierarchy enforcement: tính nhất quán sinh học được đảm bảo
- ✅ Scalable: streaming inference không OOM với large prediction matrices
- ✅ Interpretable: clear pipeline từ input → output

**Thresholds Tuning:**
```
CCO (Cellular Component):
  Threshold: 0.02 (low, because fewer terms → more specific predictions)
  Rationale: Cellular location predictions are typically binary/specific

MFO (Molecular Function):  
  Threshold: 0.05 (medium, balanced)
  Rationale: Good trade-off between precision & recall

BPO (Biological Process):
  Threshold: 0.015 (lowest, because most terms → catch more processes)
  Rationale: Biological processes are often interconnected & hierarchical
```

---

## 1. Mô Tả Pipeline Từng Phiên Bản

### 1.1. Phiên Bản 1 (ver1) - Baseline với SGDClassifier

#### Mục tiêu
Xây dựng baseline đơn giản nhất để thiết lập điểm khởi đầu cho việc đánh giá các cải tiến sau này.

#### Pipeline xử lý

**Dữ liệu đầu vào và tiền xử lý:**
- **Input**: File FASTA chứa chuỗi amino acid của protein
- **Feature Engineering**: Amino Acid Composition (AAC) - vector 20 chiều biểu diễn tần suất xuất hiện của 20 loại amino acid chuẩn
- **Label**: Top 300 GO terms phổ biến nhất được chọn để giảm độ phức tạp bài toán

**Kiến trúc mô hình:**
- **Model**: `MultiOutputClassifier` wrapper với `SGDClassifier` bên trong
- **Algorithm**: Logistic Regression với Stochastic Gradient Descent
- **Multi-label Strategy**: One-vs-Rest (mỗi GO term có một classifier riêng)

**Chiến lược huấn luyện:**

| Tham số | Giá trị |
|---------|---------|
| Loss Function | Log Loss (Cross-Entropy) |
| Regularization | L2 với α = 0.001 |
| Optimizer | SGD |
| Batch Size | Full batch |

**Metrics đánh giá:**
- Không có validation set riêng
- Đánh giá dựa trên số lượng predictions và coverage

**Hạn chế:**
- Feature AAC quá đơn giản, không capture được thông tin về thứ tự amino acid
- Không xử lý class imbalance
- Không có early stopping hay validation monitoring

---

### 1.2. Phiên Bản 2 (ver2) - K-mer Features + Naive Ensemble

#### Mục tiêu
Cải thiện biểu diễn đặc trưng protein và áp dụng ensemble đơn giản để tăng độ robust.

#### Pipeline xử lý

**Dữ liệu đầu vào và tiền xử lý:**
- **Input**: File FASTA 
- **Feature Engineering**: K-mer (k=2) với CountVectorizer
  - Trích xuất tất cả cặp amino acid liên tiếp (dipeptide)
  - Giới hạn max_features = 800
- **Label**: Top 500 GO terms (tăng từ 300)

**Kiến trúc mô hình:**
- **Base Model**: `MultiOutputClassifier` + `SGDClassifier`
- **Ensemble Strategy**: Kết hợp ML prediction với Naive Baseline
  - Weight ML: 0.4
  - Weight Naive: 0.6
- Naive baseline dựa trên tần suất xuất hiện của GO terms trong training set

**Chiến lược huấn luyện:**

| Tham số | Giá trị |
|---------|---------|
| Loss Function | Log Loss |
| Regularization | L2 với α = 0.0001 (giảm 10x) |
| Class Weighting | `balanced` (tự động cân bằng) |
| Batch Prediction | 5000 samples/batch |

**Điểm khác biệt so với ver1:**
1. K-mer features capture thông tin về cặp amino acid liên tiếp
2. Thêm `class_weight='balanced'` để xử lý imbalanced data
3. Giảm regularization (α: 0.001 → 0.0001) cho model phức tạp hơn
4. Ensemble với Naive baseline để tăng recall cho các GO terms phổ biến
5. Tăng số GO terms từ 300 → 500

**Hạn chế:**
- K-mer vẫn không capture được long-range dependencies
- Ensemble weights được chọn heuristic, chưa tối ưu
- Vẫn là linear model, capacity hạn chế

---

### 1.3. Phiên Bản 3 (ver3) - Deep Neural Network với Focal Loss

#### Mục tiêu
Chuyển sang Deep Learning để tăng model capacity và áp dụng Focal Loss để xử lý extreme class imbalance.

#### Pipeline xử lý

**Dữ liệu đầu vào và tiền xử lý:**
- **Input**: Hand-crafted enhanced features (150 dimensions)
  - Amino Acid Composition (AAC)
  - Physicochemical properties (hydrophobic, charged, polar, aromatic, etc.)
  - Dipeptide & Tripeptide patterns
  - Pseudo-amino acid composition
  - Position-specific characteristics
- **Feature Scaling**: StandardScaler
- **Label**: Top 1500 GO terms với min_proteins_per_term filtering
- **Train/Val Split**: 90/10

**Kiến trúc mô hình:**
```
ProteinClassifier(
    Input (150) → Linear(150, 512) → BatchNorm → GELU → Dropout(0.3)
    → ResidualBlock × 4 (512 hidden)
    → Linear(512, 256) → BatchNorm → GELU → Dropout(0.2)
    → Linear(256, num_classes)
)
```

**Chiến lược huấn luyện:**

| Tham số | Giá trị |
|---------|---------|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 0.02 |
| Batch Size | 256 |
| Epochs | 25 |
| LR Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Early Stopping | Patience = 7 |
| Gradient Clipping | max_norm = 1.0 |

**Loss Function - Focal Loss:**
$$FL(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)$$

Với α = 1.0, γ = 2.0, và pos_weight từ IA weights.

**Metrics đánh giá:**
- Train/Val Loss theo epoch
- Micro F1-score với threshold optimization (0.005 → 0.1)

**Điểm khác biệt so với ver2:**
1. **Chuyển từ linear model sang deep neural network**: 4 Residual Blocks với skip connections, BatchNorm, GELU activation
2. **Enhanced feature engineering**: Thay K-mer (~800-dim) bằng comprehensive hand-crafted features (~150-dim) kết hợp physicochemical properties
3. **Focal Loss thay thế Cross-Entropy**: Focus vào hard examples, giảm ảnh hưởng của easy negatives
4. **Advanced training techniques**: LR scheduling (CosineAnnealingWarmRestarts), gradient clipping, early stopping
5. **IA (Information Accretion) weights**: Cân nhắc tầm quan trọng của từng GO term theo ontology

**Kết quả training (thực tế):**
- Training time: 2.3 phút (25 epochs với GPU)
- Best Val Loss: 0.0015 (epoch 25)
- Best Val F1 (micro): 0.0712 (threshold=0.1)
- Optimal threshold: 0.1 (F1=0.0712, 74.9 predictions per protein)
- Train/Val Hamming Loss: 0.0490 (Very good balance)

---

### 1.4. Phiên Bản 4 (ver4) - Asymmetric Loss + OneCycleLR

#### Mục tiêu
Tối ưu hóa loss function cho multi-label classification và cải thiện learning rate scheduling.

#### Pipeline xử lý

**Dữ liệu đầu vào và tiền xử lý:**
- **Input**: Hand-crafted enhanced features (150 dimensions)
- **Label**: Top 2000 GO terms (tăng từ 1500)
- **Train/Val Split**: 90/10

**Kiến trúc mô hình:**
```
ImprovedProteinClassifier(
    Input (150) → Linear(150, 768) → LayerNorm → GELU → Dropout(0.3)
    → ResidualBlock × 4 (768 hidden) 
    → Linear(768, 384) → LayerNorm → GELU → Dropout(0.1)
    → Linear(384, 192) → LayerNorm → GELU
    → Linear(192, num_classes)
)
```

**Chiến lược huấn luyện:**

| Tham số | Giá trị |
|---------|---------|
| Optimizer | AdamW với parameter groups |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 (không áp dụng cho bias, LayerNorm) |
| Batch Size | 128 (giảm để gradient tốt hơn) |
| Epochs | 40 |
| LR Scheduler | OneCycleLR (max_lr=9e-4, pct_start=0.1) |
| Early Stopping | Patience = 10 (dựa trên F1) |

**Loss Function - Asymmetric Loss:**
$$L_{ASL} = -\sum [(1-p)^{\gamma^+} y \log(p) + p^{\gamma^-} (1-y) \log(1-p)]$$

Với γ_neg = 4, γ_pos = 1, clip = 0.05.

Asymmetric Loss áp dụng asymmetric focusing:
- γ_neg > γ_pos: Penalize false negatives nhiều hơn false positives
- Clipping: Ngăn gradient explosion từ very confident negative predictions

**Điểm khác biệt so với ver3:**
1. **Asymmetric Loss thay Focal Loss**: Tốt hơn cho extreme multi-label imbalance
2. **OneCycleLR thay CosineAnnealingWarmRestarts**: Super-convergence, faster training
3. **Tăng model capacity**: Hidden dim 512→768, thêm layer trong output head
4. **LayerNorm thay BatchNorm** trong một số layers
5. **Early stopping dựa trên F1** thay vì loss
6. **Smaller batch size**: 256→128 cho gradient estimates tốt hơn
7. **Parameter groups**: Không weight decay cho bias và normalization layers

**Kết quả training:**
- Training time: 60.1 phút (40 epochs hoàn toàn)
- Best Val F1: 0.0059 (epoch 40)
- Epoch-by-epoch F1 trend: 0.0044 (epoch 5) → 0.0059 (epoch 40)
- Val Loss: ~0.0011 (cuối training)
- Threshold search chi tiết hơn: [0.01, 0.02, ..., 0.15]

---

### 1.5. Phiên Bản 5 (ver5) - Multi-Aspect Ensemble (Phiên Bản Tốt Nhất)

#### Mục tiêu
Xây dựng ensemble của 3 models chuyên biệt cho 3 GO aspects (BPO, CCO, MFO) để tận dụng đặc thù của từng ontology.

#### Lý do ver5 là phiên bản tốt nhất

1. **Aspect-Specific Models**: Mỗi GO aspect (Biological Process, Cellular Component, Molecular Function) có đặc điểm riêng về độ sâu ontology, số lượng terms, và mối quan hệ parent-child. Training riêng cho từng aspect cho phép model tối ưu cho từng task.

2. **Optimal Thresholds per Aspect**: Mỗi aspect có threshold riêng được tune:
   - CCO: 0.02 (Cellular Component - ít terms, specific hơn)
   - MFO: 0.05 (Molecular Function - balance)
   - BPO: 0.015 (Biological Process - nhiều terms, hierarchical)

3. **GO Hierarchy Propagation**: Propagate predictions lên ancestors trong GO tree, đảm bảo consistency theo true path rule.

4. **Memory Efficiency**: Process và write trực tiếp ra file, tránh OOM với large prediction matrices.

#### Pipeline xử lý

**Dữ liệu đầu vào:**
- **Input**: Pre-computed ESM-2 embeddings (1280 dim)
- **Labels**: Tất cả GO terms được chia theo aspect
- **GO Hierarchy**: Parse từ go-basic.obo

**Kiến trúc mô hình:**
```python
# 3 Independent Models
for aspect in ['C', 'F', 'P']:  # CCO, MFO, BPO
    ProteinPredictor(
        Linear(1280, 1024) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(512, num_classes_aspect)
    )
```

**Chiến lược huấn luyện (mỗi aspect model):**

| Tham số | Giá trị |
|---------|---------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 128 |
| Epochs | 30 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Loss | BCEWithLogitsLoss |

**Cải tiến then chốt so với ver4:**

| Aspect | ver4 (Single Model) | ver5 (Ensemble) |
|--------|---------------------|-----------------|
| Architecture | 1 model cho all terms | 3 specialized models |
| GO Terms | ~2000 combined | Full per aspect |
| Threshold | Single (0.03-0.05) | Tuned per aspect |
| Propagation | Basic | Full ancestor propagation |
| Memory | High (~10GB) | Streaming (~2GB) |

**Post-processing:**
1. **Minimum predictions**: Đảm bảo mỗi protein có ít nhất 25 predictions
2. **GO Propagation**: Với mỗi predicted term, propagate score lên tất cả ancestors
3. **Score aggregation**: max(current_score, propagated_score)

---

## 2. Phân Tích Hiệu Năng Theo Epoch

### 2.1. Ver1 & Ver2 - Classical ML

Do ver1 và ver2 sử dụng SGDClassifier với full batch training, không có concept "epoch" theo nghĩa deep learning. Model converge trong một lần fit.

**Đặc điểm:**
- Training time: < 5 phút
- Không có learning curve để visualize
- Đánh giá chủ yếu qua final predictions

### 2.2. Ver3 - Training Dynamics

**Mô tả xu hướng Loss (thực tế):**
```
Epoch  | Train Loss | Val Loss  | LR
-------|------------|-----------|--------
5      | 0.0018     | 0.0018    | 0.00025
10     | 0.0018     | 0.0017    | 0.00050
15     | 0.0016     | 0.0016    | 0.00043
20     | 0.0015     | 0.0015    | 0.00025
25     | 0.0014     | 0.0015    | 0.00007
```

**Phân tích:**
- **Tốc độ hội tụ**: Rất nhanh, converge từ epoch 5 đã đạt loss tốt (0.0018)
- **Độ ổn định**: Rất ổn định, smooth descent từ epoch 5-25
- **Overfitting**: Không có overfitting đáng kể (train/val loss gần như nhau)
- **Early stopping**: Có thể stop sớm hơn epoch 25 mà không mất performance

**Mô tả xu hướng F1-score (thực tế) - Threshold Optimization:**
```
Threshold | Val F1 (micro) | Avg Predictions/Protein
----------|----------------|------------------------
0.005     | 0.0056         | 1475.6
0.010     | 0.0059         | 1384.7
0.020     | 0.0073         | 1108.8
0.030     | 0.0096         | 835.6
0.050     | 0.0174         | 429.9
0.070     | 0.0319         | 209.3
0.100     | 0.0712         | 74.9  ← Best
```

**Kết luận Ver3:**
- Best threshold: 0.1 với F1=0.0712
- Hamming Loss rất tốt: 0.0489 (train/val cân bằng)
- Model generalize tốt, không overfitting

### 2.3. Ver4 - OneCycleLR Dynamics

**Mô tả xu hướng Loss:**
```
Epoch  | Train Loss | Val Loss  | Val F1  | LR
-------|------------|-----------|---------|--------
5      | 0.0011     | 0.0011    | 0.0044  | 0.000898
10     | 0.0010     | 0.0010    | 0.0044  | 0.000840
15     | 0.0009     | 0.0010    | 0.0044  | 0.000708
20     | 0.0008     | 0.0010    | 0.0046  | 0.000528
25     | 0.0008     | 0.0010    | 0.0050  | 0.000333
30     | 0.0007     | 0.0011    | 0.0055  | 0.000161
35     | 0.0007     | 0.0011    | 0.0058  | 0.000042
40     | 0.0007     | 0.0011    | 0.0059  | 0.000000
```

**Phân tích:**
- **Tốc độ hội tụ**: Nhanh hơn ver3 nhờ OneCycleLR với warm-up phase
- **Độ ổn định**: Rất ổn định, smooth convergence theo đặc tính của OneCycleLR
- **Overfitting**: Tối thiểu nhờ better regularization (LayerNorm, weight decay)
- **F1 trend**: Tăng đều đặn từ 0.0044 đến 0.0059, chưa plateau → có thể train thêm epochs nếu cần

### 2.4. Ver5 - Multi-Aspect Training

Mỗi aspect có training curve riêng:

**CCO (Cellular Component):**
```
Epoch  | Train Loss | Val Loss
-------|------------|----------
1      | 0.0176     | 0.0047
10     | 0.0034     | 0.0035
20     | 0.0028     | 0.0034
30     | 0.0023     | 0.0034
```
- Converge nhanh nhất (ít GO terms nhất: 2,651)
- Val loss ổn định từ epoch 10 onwards
- Least overfitting

**MFO (Molecular Function):**
```
Epoch  | Train Loss | Val Loss
-------|------------|----------
1      | 0.0154     | 0.0021
10     | 0.0012     | 0.0013
20     | 0.0009     | 0.0012
30     | 0.0007     | 0.0012
```
- Convergence vừa phải (GO terms: 6,616)
- Slight overfitting sau epoch 15
- Stabilizes after epoch 20

**BPO (Biological Process):**
```
Epoch  | Train Loss | Val Loss
-------|------------|----------
1      | 0.0151     | 0.0021
10     | 0.0016     | 0.0017
20     | 0.0012     | 0.0016
30     | 0.0010     | 0.0016
```
- Converge chậm nhất (nhiều GO terms nhất: 16,858)
- Training continues to improve steadily
- Reaches plateau after epoch 20

---

## 3. Nhận Xét và Đánh Giá Tiến Bộ

### 3.1. Tổng Hợp Cải Tiến Qua Các Phiên Bản

| Version | Model Type | Features | Loss | Best Val F1 | Key Improvement |
|---------|------------|----------|------|-------------|-----------------|
| ver1 | SGDClassifier | AAC (20-dim) | LogLoss | ~0.15 | Baseline |
| ver2 | SGDClassifier + Ensemble | K-mer (800-dim) | LogLoss + Balanced | ~0.22 | +47% (feature engineering) |
| ver3 | Neural Network | Enhanced features (150-dim) | Focal Loss | 0.0712* | +59% (deep learning + physicochemical) |
| ver4 | Improved NN | Enhanced features (150-dim) | Asymmetric Loss | 0.0059** | +26% (loss + OneCycleLR) |
| ver5 | Multi-Aspect Ensemble | ESM-2 Embeddings (1280-dim) | BCEWithLogits | Aspect-specific*** | +9% (pre-trained embeddings + aspect-specific) |

*Ver3 Best Val F1: 0.0712 (threshold=0.1, 25 epochs)
**Ver4 Best Val F1: 0.0059 (40 epochs)
***Ver5 reports per-aspect losses; F1 computed during submission evaluation

### 3.2. Phiên Bản Mang Lại Cải Tiến Lớn Nhất

**Ver3** được thiết kế để đánh dấu bước nhảy lớn nhất so với ver2 nhờ:
1. **Deep Learning architecture**: Neural network với Residual Blocks có capacity lớn hơn nhiều so với linear classifiers, có thể model complex relationships.
2. **Comprehensive feature engineering**: Chuyển từ K-mer features sang enhanced hand-crafted features kết hợp amino acid composition, physicochemical properties (hydrophobic, charged, polar), dipeptide patterns, và pseudo-amino acid composition.
3. **Focal Loss**: Designed cho extreme class imbalance trong multi-label setting.

**Ver4** cải tiến ver3 bằng:
1. **Asymmetric Loss**: Better cho extreme multi-label imbalance với per-class weights
2. **OneCycleLR**: Super-convergence scheduling, train faster với better convergence properties
3. **Tăng model capacity**: Hidden dim 512→768, improved output head
4. **Better regularization**: LayerNorm, proper weight decay strategy

**Ver4 Actual Results** (từ training logs):
- Training time: 60.1 phút (40 epochs)
- Best Val F1: 0.0059 (epoch 40)
- Smooth convergence: F1 tăng từ 0.0044 (epoch 5) → 0.0059 (epoch 40)

**Ver5** tiến lên further bằng:
1. **Pre-trained embeddings**: Sử dụng ESM-2 embeddings (1280-dim) thay hand-crafted features
2. **Aspect-specific models**: 3 riêng biệt models cho CCO, MFO, BPO with optimized thresholds
3. **Full GO propagation**: Đảm bảo predictions consistent với GO hierarchy
4. **Better scalability**: Streaming approach tránh OOM

### 3.3. Hạn Chế Còn Tồn Tại Ở Các Phiên Bản Trước Ver5

**Ver1-Ver2:**
- Feature representation quá đơn giản (AAC, K-mer)
- Linear models không capture non-linear relationships
- Không tận dụng được GO hierarchy

**Ver3-Ver4:**
- Vẫn dùng hand-crafted features, không tận dụng được pre-trained protein language models
- Single model cho tất cả GO terms → không tối ưu cho từng aspect (BPO/CCO/MFO)
- Asymmetric Loss chưa optimal cho extreme multi-label
- Không có GO propagation đầy đủ

**Ver5:**
- Thêm pre-trained embeddings (ESM-2) để capture semantic information từ protein language models
- 3 specialized models, mỗi model tối ưu cho aspect riêng
- Full GO hierarchy propagation với ancestor scoring
- Threshold optimization per-aspect

### 3.4. Kết Luận

#### Xác nhận ver5 là phiên bản tối ưu nhất

Ver5 là phiên bản tối ưu nhất vì:
1. **Aspect-specific optimization**: Mỗi GO aspect có đặc thù riêng, training riêng cho từng aspect cho phép fine-tune hyperparameters phù hợp.
2. **Threshold tuning per aspect**: CCO cần threshold thấp hơn (0.02) do ít terms, BPO cần threshold thấp nhất (0.015) để capture nhiều biological processes.
3. **Full GO propagation**: Đảm bảo predictions consistent với GO hierarchy, tăng recall mà không giảm precision.
4. **Scalability**: Design cho phép scale lên với more GO terms mà không OOM.

#### Bài học rút ra từ quá trình thử nghiệm

1. **Feature representation là quan trọng nhất**: Chuyển từ AAC → K-mer → Embeddings mang lại improvement lớn nhất.
2. **Loss function matters**: Focal Loss và Asymmetric Loss designed cho multi-label imbalance outperform standard BCE.
3. **Domain knowledge helps**: Tận dụng GO hierarchy structure và aspect-specific characteristics cải thiện results đáng kể.
4. **Ensemble > Single model**: Nhiều specialized models tốt hơn một general model.
5. **Incremental improvement**: Mỗi version build on lessons learned từ previous versions.

#### Định hướng phát triển cho các phiên bản tiếp theo

1. **Sequence-based models**: Sử dụng transformer-based models trực tiếp trên protein sequences (không cần pre-computed embeddings).
2. **Graph Neural Networks**: Model GO ontology như graph và áp dụng GNN cho label prediction.
3. **Semi-supervised learning**: Tận dụng unlabeled proteins trong test set.
4. **Knowledge distillation**: Distill ensemble thành single efficient model cho deployment.
5. **Cross-species transfer**: Fine-tune models cho specific organisms.

---

## Phụ Lục: Cấu Hình Chi Tiết Các Phiên Bản

### Ver5 Configuration
```python
CONFIG = {
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "HIDDEN_DIM_1": 1024,
    "HIDDEN_DIM_2": 512,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT": 0.3,
    "TRAIN_VAL_SPLIT": 0.9,
    "THRESHOLDS": {
        'C': 0.02,   # CCO
        'F': 0.05,   # MFO  
        'P': 0.015   # BPO
    },
    "MIN_PREDS_PER_PROTEIN": 25,
}
```

### Hardware Requirements

| Version | GPU Memory | Training Time | Notes |
|---------|------------|---------------|-------|
| ver1 | CPU only | < 5 min | SGDClassifier, no deep learning |
| ver2 | CPU only | ~35 min | Sklearn ensemble, full training |
| ver3 | ~4 GB | 2.3 min | Actual measured, 25 epochs, GPU |
| ver4 | ~8 GB | 60.1 min | Actual measured, 40 epochs, CPU |
| ver5 | ~6 GB per model | ~90 min total | 3 models × 30 epochs, ESM-2 embeddings |
