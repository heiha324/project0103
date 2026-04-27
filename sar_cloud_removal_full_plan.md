# SAR 引导条件扩散去云（带自然边缘过渡）完整方案（Cloud-only 云检 + RS-Net + SEN12MS-CR）

> 设定：
> - 云检测：Sentinel-2 云检数据集（单样本 384×384），**只有 cloud 标注**，网络用 **RS-Net**
> - 去云：**SEN12MS-CR**（S1 SAR + cloudy S2 + cloud-free S2，patch 为 256×256）
> - 目标：输出去云后的 S2（多波段/或 RGB），**clear 区尽量保真**，云区补全自然，**云边缘有平滑过渡**（避免硬拼接、halo、暗区裂缝）

---

## 1. 总体流水线（两阶段 + 缓存桥接）

### 阶段 A：云检测（RS-Net）
输入：S2(可选加 SWIR) → 输出：云概率图  
- 输出：`p_cloud ∈ [0,1]`（单头 cloud-only）

### 阶段 B：α 构造（关键桥梁：自然过渡 + shadow 安全）
从 `p_cloud` 构造 “观测可信度羽化权重”  
- 定义：`p_clear = 1 - p_cloud`
- 构造：`α ∈ [0,1]`（α 越大越“钉住观测”，越小越允许生成）
- 增强：云邻域降可信度（shadow safety），避免云影被当 clear 钉死

### 阶段 C：对 SEN12MS-CR 全量生成并缓存 α
对每个 SEN12MS-CR 的 cloudy S2 patch：
- 跑 RS-Net → 得 p_cloud
- 构造 α
- 缓存 α（uint8/float16），供扩散训练/推理直接读取

### 阶段 D：SAR 引导条件扩散去云
训练：条件扩散模型学习 `p(x | y, s, α)`  
推理：**软 data consistency** + **时间日程**实现自然边缘过渡

---

## 2. 数据与预处理（统一规范，避免域漂）

### 2.1 Sentinel-2（S2）
**建议波段：**
- 最低可用：B2,B3,B4,B8（RGB+NIR）
- 更稳（推荐）：+ B11,B12（SWIR，上采样到同分辨率）

**数值处理：**
- clip：按数据集实际范围（常见 0~10000 或 0~1）
- 归一化：统一到 [0,1] 或标准化（推荐：clip 后除以 10000 → [0,1]）

### 2.2 Sentinel-1（S1, SAR）
- 转 dB：`10*log10(x + eps)`
- clip（经验范围）：如 [-25, 0] dB（根据数据统计可调）
- 归一化到 [0,1]：`(val - min)/(max-min)`

> 关键：所有阶段（云检训练/云检推理/扩散训练/扩散推理）尽量使用一致的归一化方式。

---

## 3. 阶段 A：云检数据切割（384×384 → 256×256）【你要求“云检就切割”】【重要：别只取左上角】

### 3.1 基础裁剪：滑窗 4 块（stride=128）
对每个 384×384 样本取 4 个 256×256：
- (0,0)、(0,128)、(128,0)、(128,128)

### 3.2 随机抖动（强烈建议）
每个 crop 加随机偏移（每个 epoch 重采样即可）：
- `dx, dy ~ Uniform[-16, +16]`
- clamp 到合法范围 `x∈[0,128], y∈[0,128]`

目的：
- 避免固定网格位置偏置
- 增强样本多样性，提高泛化

### 3.3 云比例过滤（只有 cloud 标签）
对每个 crop 计算：
- `cloud_ratio = cloud_pixels / (256*256)`

保留规则（默认）：
- 必留：`cloud_ratio ≥ 1%`
- 否则：以 `20%` 概率保留（hard negatives）

### 3.4 Batch 采样均衡（强烈建议）
分桶：
- cloud-heavy：`cloud_ratio ≥ 10%`
- cloud-light：`1% ≤ cloud_ratio < 10%`
- clear：`cloud_ratio < 1%`

每 batch 比例：
- `40% heavy + 40% light + 20% clear`

### 3.5 数据增强（同步影像与 mask）
- 随机水平翻转
- 随机垂直翻转
- 随机 90°/180°/270° 旋转
-（可选）轻微亮度/对比度（幅度小，避免破坏物理反射率分布）

---

## 4. 阶段 A：RS-Net（cloud-only）训练细节

### 4.1 输入 / 输出
- 输入：S2 多通道（建议至少 RGB+NIR，最好加 SWIR）
- 输出：单头 cloud logit → sigmoid → `p_cloud`

### 4.2 损失函数（处理不均衡 + 边界质量）
推荐总损失：
- `BCEWithLogits(pos_weight=w)` + `DiceLoss`

参数建议：
- `w = 3 ~ 8`（云越少 w 越大；>10 容易把边界抹粗）
- `λ_dice = 1.0`

（可选）边界加权 BCE：
- 对云 mask 做形态学梯度得到“边界带”
- 边界像素 BCE 权重 ×2（提升边界精细度）

### 4.3 训练超参（默认可直接跑）
- patch：256×256
- batch：16（不足则 8 + 梯度累积）
- optimizer：AdamW
- lr：1e-3
- weight_decay：1e-4
- epochs：50–100
- AMP：开启
- grad clip：可设 1.0

### 4.4 验证与保存策略
建议同时监控：
- IoU（cloud）
- Precision / Recall（云类）
- FPR（clear 上误检率）

模型保存推荐“综合评分”而非只看 IoU：
- `score = IoU + 0.5*Recall - 0.25*FPR`（可按偏好调整）

---

## 5. 概率校准（必须：后面 α 会把概率当权重）

### 5.1 温度缩放（推荐最简单稳定）
在验证集上拟合单标量温度 `T`：
- `p = sigmoid(logit / T)`
- 用 NLL/BCE 最小化拟合 T

输出：
- `T`（保存到配置文件）

---

## 6. 阶段 B：从 p_cloud 构造 α（自然过渡 + shadow safety）【核心】

> 目标：α 表示“观测可信度”
> - clear 核心：α≈1（尽量不改）
> - 云区：α≈0（允许生成）
> - 边界：α 平滑从 1→0（自然过渡）
> - 云邻域（潜在云影/薄云/污染区）：降低 α（防止暗区裂缝、边缘撕裂）

### 6.1 基础 clear 概率
- `p_clear = 1 - p_cloud`

### 6.2 可控锐化（让云/非云分界更干净）
用 sigmoid 锐化：
- `α0 = sigmoid( k * (p_clear - τ) )`

默认参数：
- `τ = 0.5`
- `k = 12`（想更硬：16；更软：8）

### 6.3 羽化（让边缘自然渐变）
对 α0 做高斯模糊：
- `α = GaussianBlur(α0, sigma=1.5 px)`
- clamp 到 [0,1]

> sigma 太小：边缘偏硬；太大：clear 区会被“放开”导致漂色。建议 1~2 起步。

### 6.4 Shadow safety（只有 cloud 标签时必须！）
#### 6.4.1 云邻域降可信度（默认强烈推荐）
1) 云核：
- `cloud_core = (p_cloud > 0.6)`

2) 云邻域环：
- `cloud_ring = dilate(cloud_core, radius=8 px)`

3) 降权：
- `α = α * (1 - 0.5 * cloud_ring)`

默认：
- radius = 8（256×256 patch 很合适；厚云多可 12）
- 0.5（想更保守可 0.6；更保真可 0.4）

#### 6.4.2（可选增强）云邻域内暗像素再降一次（更像云影）
仅在 cloud_ring 内判断暗像素，避免误伤水体/暗地物：
- `dark = mean(R,G,B,NIR) < P10`（patch 内 10% 分位数）
- `shadow_suspect = dark & cloud_ring`
- `α = α * (1 - 0.7 * shadow_suspect)`

### 6.5 m_hard（用于最终保底）
- `m_hard = (α > 0.97)`（可调：0.985 更严格）

---

## 7. 阶段 C：在 SEN12MS-CR 上生成并缓存 α（工程化细节）

### 7.1 离线缓存（强烈建议）
对 SEN12MS-CR 的每个 cloudy patch：
1) 读取 cloudy S2（按云检一致的预处理）
2) RS-Net 输出 logit → 用温度 T 校准 → 得 p_cloud
3) 按第 6 节构造 α
4) 保存：
- 推荐：uint8 `alpha_u8 = round(255*α)`（省空间、速度快）
- 或 float16（更精细）

目录建议：
```
alpha_cache/
  train/*.png or *.npy
  val/*.png or *.npy
  test/*.png or *.npy
```

---

## 8. 阶段 D：SAR 引导条件扩散去云（训练细节全量）

### 8.1 训练目标
学习条件分布：
- 输入条件：`(y = cloudy S2, s = SAR S1, α)`
- 输出：`x0 = cloud-free S2`

### 8.2 网络结构（推荐实现）
**三模块：**
1) Opt+α Encoder：输入 `[y ⊕ α]` → 多尺度特征 `f_o`
2) SAR Encoder：输入 `s` → 多尺度特征 `f_s`
3) Denoiser U-Net：输入 `x_t` + time embedding → 融合 `f_o, f_s` → 输出 `ε_pred`（或 v_pred）

**融合建议（稳定且效果好）：**
- 多尺度注入：在 U-Net 每个分辨率层注入条件特征
- FiLM（scale-shift）+ 门控：
  - clear 区（α高）抑制 SAR 影响
  - 云区（α低）增强 SAR 引导

### 8.3 扩散设置（建议起步）
- 预测目标：ε-pred（实现简单稳定）
- schedule：cosine 或 linear（cosine 常更稳）
- 训练 T：1000
- 推理步数：30（薄云）/ 50（厚云）

当前实现补充：
- `scripts/train_diffusion_rs_transformer_v3.py` 是 Transformer 主干的标准 DDPM 版本，只替换 v2 的扩散过程，保留模型结构、loss、EMA、日志和 checkpoint 保存逻辑。
- v3 前向加噪公式：`x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * ε`。
- v3 模型仍使用条件输入 `(x_t, t, y, s1)`，其中 `y` 是有云 S2，`s1` 是 SAR；采样默认从纯高斯噪声开始。

### 8.4 损失函数（必须：扩散 + 区域化约束）
1) 扩散噪声损失：
- `L_diff = MSE(ε, ε_pred)`

2) clear 保真（按 α 加权）：
- `L_id = L1( α ⊙ (x_hat0 - y) )`

3) 云区重建（按 1-α 加权，有真值 x0）：
- `L_cloud = L1( (1-α) ⊙ (x_hat0 - x0) )`

4) 边界梯度连续（治 halo/接缝关键）：
- `b = α(1-α)`
- `L_grad = L1( b ⊙ (∇x_hat0 - ∇y) )`（∇ 用 Sobel/Laplacian）

总损失默认：
- `L = L_diff + 1.0*L_id + 1.0*L_cloud + 0.5*L_grad`

调参口诀：
- clear 漂色：`L_id` ↑ 或 `m_hard` 阈值 ↑
- 边缘接缝/halo：`L_grad` ↑、α blur sigma ↑
- 云区补不准：`L_cloud` ↑ 或增加采样步数

### 8.5 训练超参（默认可直接跑）
- batch：8–16
- optimizer：AdamW
- lr：1e-4
- weight_decay：1e-4
- EMA：0.999
- AMP：开
- grad clip：1.0

---

## 9. 推理（采样）：软 data consistency + 时间日程（自然过渡最终落地）

### 9.1 软 data consistency（每一步都做）
每一步先用采样器得到 `x'_{t-1}`，再融合观测：
- `x_{t-1} = α_t ⊙ noised(y, t-1) + (1-α_t) ⊙ x'_{t-1}`

> 注意：必须用 `noised(y, t-1)`，不能直接用 y（噪声域不一致会破坏采样）。

### 9.2 时间日程（早期松、后期紧）
- `α_t = α * w(t)`

默认 w(t)（N 步）：
- 前 40%：w = 0.2
- 中 40%：w 线性升到 0.8
- 后 20%：w 线性升到 1.0

### 9.3 最终保底（强烈建议）
- `m_hard = (α > 0.97)`
- `x_hat0 = m_hard ⊙ y + (1-m_hard) ⊙ x_hat0`

效果：
- clear 核心几乎像素级不动
- 过渡带自然渐变
- 云区自由生成（受 SAR 引导）

### 9.4（可选）不确定性图
同一输入多次采样（不同随机种子）：
- 均值：最终输出
- 方差：不确定性（云越厚通常越高）

---

## 10. 评估（必须分区，不要只算整图）

用 α 定义三个区域：
- 云区：`1-α`（或 α<0.3）
- 边界带：`b=α(1-α)`（或 0.3≤α≤0.7）
- clear 区：α>0.9

分别计算：
- 云区：PSNR/SSIM/L1（核心）
- 边界带：梯度误差/SSIM（衡量过渡）
- clear 区：`|x_hat - y|`（保真度）

建议按云量分桶（mean(1-α)）：
- 0–10%、10–30%、30–60%、60–100%
分别报云区指标（厚云表现更有说服力）

---

## 11. 必做消融（写论文/定位问题必备）

至少 6 个：
1) 不用 SAR：条件只用 (y, α)
2) 不用 α：条件 (y, s)，推理不做软一致性
3) 硬一致性（α阈值化成0/1）vs 软一致性（本方案）
4) 去掉 shadow safety ring（第 6.4）看云影/暗区是否崩
5) 去掉 L_grad 看 halo/接缝恶化
6) 去掉时间日程（w(t)=1 固定）看早期被钉死导致生成受限

---

## 12. 常见问题与快速修复

### 12.1 clear 区漂色
- L_id：1.0→2.0
- m_hard 阈值：0.97→0.985
- α blur sigma 降一点（避免 clear 被羽化放开）

### 12.2 边缘接缝/halo
- L_grad：0.5→1.0
- α blur sigma：1.5→2.0
- w(t) 前期更小（0.2→0.1），后期更大（更贴观测）

### 12.3 云影暗区裂缝
- cloud_ring 半径：8→12
- 降权系数：0.5→0.6
- 启用“暗像素 AND cloud_ring”增强降权

### 12.4 厚云结构乱
- 推理步数：30→50
- 增强 SAR 门控（α低时 SAR 更强）
- 检查 SAR dB clip 范围是否过窄导致信息丢失

---

## 13. 伪代码（可直接照着写）

### 13.1 384→256 切割 + 抖动 + 过滤
```python
def crops_384_to_256(img384, mask384):
    bases = [(0,0), (0,128), (128,0), (128,128)]
    crops = []
    for (x,y) in bases:
        dx = randint(-16, 16)
        dy = randint(-16, 16)
        xx = clamp(x+dx, 0, 128)
        yy = clamp(y+dy, 0, 128)
        crop_img  = img384[:, yy:yy+256, xx:xx+256]
        crop_mask = mask384[yy:yy+256, xx:xx+256]
        cloud_ratio = crop_mask.mean()
        if cloud_ratio >= 0.01 or rand() < 0.2:
            crops.append((crop_img, crop_mask, cloud_ratio))
    return crops
```

### 13.2 p_cloud → α（锐化 + 羽化 + shadow safety）
```python
def build_alpha(p_cloud, rgbnir=None):
    p_clear = 1.0 - p_cloud

    # sharpen
    tau, k = 0.5, 12
    alpha0 = sigmoid(k * (p_clear - tau))

    # feather
    alpha = gaussian_blur(alpha0, sigma=1.5).clip(0,1)

    # shadow safety ring
    cloud_core = (p_cloud > 0.6).astype(np.uint8)
    cloud_ring = dilate(cloud_core, radius=8)
    alpha = alpha * (1.0 - 0.5 * cloud_ring)

    # optional: dark pixels inside ring
    if rgbnir is not None:
        dark = (rgbnir.mean(axis=0) < np.percentile(rgbnir.mean(axis=0), 10))
        shadow_suspect = dark & (cloud_ring > 0)
        alpha = alpha * (1.0 - 0.7 * shadow_suspect.astype(np.float32))

    return alpha.clip(0,1)
```

### 13.3 推理：软 data consistency + 时间日程 + 最终保底
```python
def schedule_w(step, N):
    if step < 0.4*N: return 0.2
    if step < 0.8*N:
        return 0.2 + (step-0.4*N) * (0.6/(0.4*N))
    return 0.8 + (step-0.8*N) * (0.2/(0.2*N))

x_t = randn_like(y)
for step, t in enumerate(reversed(range(N))):
    eps = unet(x_t, t, cond=(y, s, alpha))
    x_prev_prime = sampler_step(x_t, eps, t)

    w = schedule_w(step, N)
    alpha_t = alpha * w

    y_noised = add_noise(y, t-1)
    x_t = alpha_t * y_noised + (1 - alpha_t) * x_prev_prime

x_hat = x_t
m_hard = (alpha > 0.97).astype(np.float32)
x_hat = m_hard * y + (1 - m_hard) * x_hat
```

---

## 14. 默认参数表（起步就能跑）

| 模块 | 参数 | 默认 |
|---|---|---|
| 云检切割 | stride | 128 |
| 云检切割 | jitter | ±16 |
| 云检过滤 | cloud_ratio 必留阈值 | 1% |
| 云检过滤 | clear 采样概率 | 20% |
| RS-Net loss | pos_weight | 3–8 |
| RS-Net loss | λ_dice | 1.0 |
| α 锐化 | τ | 0.5 |
| α 锐化 | k | 12 |
| α 羽化 | blur sigma | 1.5 px |
| shadow safety | p_cloud 阈值 | 0.6 |
| shadow safety | ring 半径 | 8 px |
| shadow safety | 降权系数 | 0.5 |
| m_hard | 阈值 | 0.97 |
| 扩散损失 | L_id | 1.0 |
| 扩散损失 | L_cloud | 1.0 |
| 扩散损失 | L_grad | 0.5 |
| 推理步数 | N | 30/50 |
| w(t) | 前/中/后 | 0.2→0.8→1.0 |

---

## 15. 最终交付清单（你项目里应当有的文件）

- `rsnet_cloud_only.pth`
- `rsnet_calibration.json`（温度 T）
- `alpha_cache/{train,val,test}/...`（uint8/float16）
- `sar_guided_diffusion.pth`
- `sar_guided_diffusion_ema.pth`
- `configs/`（记录所有参数，保证可复现）
- `eval/`（分区评估脚本 + 消融配置）
