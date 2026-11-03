# BitFit 在 MRPC 上的中期进展

本组检验两个问题：开放预训练 BERT 的偏置后，能否用少量可训练参数接近完整微调？如果进一步只开放注意力 query 的偏置，性能会损失多少？

## 原文位置与本组设置

原文第 3 节定义 BitFit，第 4 节表 2 的 BB 行和表 3 的 MRPC 列可直接定位本任务。原文表 3 报告完整微调、BitFit、query 子集、冻结编码器的验证 F1 分别为 89.0、90.4、84.4、81.1。附录 A.1 对照 bias 名称，A.2、A.3 及表 5、6 给出优化设置和指标。[论文第 4 页](https://aclanthology.org/2022.acl-short.1.pdf#page=4)

本组固定 BERT-base-uncased，只做 MRPC。原文用 batch 16、5 个种子并搜索学习率，表 6 的 BERT-base MRPC 学习率为 7e-4。本组用 batch 8、3 个种子、固定 3 轮和预设学习率，报告最终验证结果。表中计数包含完整分类头。仅 query 的消融冻结 pooler bias。环境、资源版本、样本 ID 与运行命令见 [README](../README.md)。

## 本组验证结果

已完成种子 17、42、73。每次用 3,668 条训练样本训练三轮，在相同的 408 条验证样本上评价。分数为百分数。误差项为三个种子的样本标准差。

| 方法 | F1 | Accuracy | 可训练参数 | 峰值显存 MiB¹ | 训练秒数¹ |
|---|---:|---:|---:|---:|---:|
| 完整微调 | 90.87 ± 0.68 | 87.09 ± 0.86 | 109,483,778 | 2155.3 | 109.1 |
| 仅分类头 | 81.05 ± 0.00 | 68.14 ± 0.00 | 1,538 | 479.9 | 21.9 |
| BitFit | 88.80 ± 1.17 | 83.91 ± 1.97 | 104,450 | 1013.7 | 48.3 |
| 仅 query bias + 分类头 | 81.03 ± 0.15 | 69.85 ± 0.25 | 10,754 | 999.6 | 39.8 |

¹ 为已完成种子的均值。耗时仅含训练；峰值取 PyTorch allocated。逐次损失、耗时和显存保存在各组的 metrics.json。

在这组设置下，BitFit 的平均 F1 比完整微调低 2.07 分，比仅分类头高 7.75 分。只开放 query bias 使 F1 比 BitFit 低 7.77 分，峰值显存却只减少约 14 MiB。可训练参数的减少没有带来同比例的显存节省。这些结果限于本次固定训练预算和 MRPC 验证集。
 
### 逐条预测与审计

| 种子 | 完整微调 | 仅分类头 | BitFit | 仅 query bias + 分类头 |
|---|---|---|---|---|
| 17 | [预测](../results/17/full/predictions.jsonl) · [梯度与差值](../results/17/full/audit.json) | [预测](../results/17/head/predictions.jsonl) · [梯度与差值](../results/17/head/audit.json) | [预测](../results/17/bitfit/predictions.jsonl) · [梯度与差值](../results/17/bitfit/audit.json) | [预测](../results/17/query/predictions.jsonl) · [梯度与差值](../results/17/query/audit.json) |
| 42 | [预测](../results/42/full/predictions.jsonl) · [梯度与差值](../results/42/full/audit.json) | [预测](../results/42/head/predictions.jsonl) · [梯度与差值](../results/42/head/audit.json) | [预测](../results/42/bitfit/predictions.jsonl) · [梯度与差值](../results/42/bitfit/audit.json) | [预测](../results/42/query/predictions.jsonl) · [梯度与差值](../results/42/query/audit.json) |
| 73 | [预测](../results/73/full/predictions.jsonl) · [梯度与差值](../results/73/full/audit.json) | [预测](../results/73/head/predictions.jsonl) · [梯度与差值](../results/73/head/audit.json) | [预测](../results/73/bitfit/predictions.jsonl) · [梯度与差值](../results/73/bitfit/audit.json) | [预测](../results/73/query/predictions.jsonl) · [梯度与差值](../results/73/query/audit.json) |

每组检查全部 201 个参数张量。每个冻结张量的首步梯度为空、训练前后逐元素完全一致。BitFit 的 104,450 个可训练参数约占全模型 0.0954%。可训练参数量指进入优化器的元素总数，每个张量实际发生数值变化的元素数另存于审计文件。

### 两个失败案例

对种子 17 的 BitFit 误判各选一条假阳性和假阴性，以下概率均为正类概率。可用 idx 从固定验证集还原英文原句。

- **idx 284，谁替谁作证？** 第一句预计 Muhammad 会在 Malvo 的听证会上出庭作证，第二句说 Malvo 会在 Muhammad 的听证会上作证。人名和事件相同，但角色互换，金标为 0。BitFit 给出 0.723，完整微调给出 0.217。这一例检验模型能否区别词汇重合与事件角色一致。[BitFit 输出第 27 行](../results/17/bitfit/predictions.jsonl#L27) · [完整微调](../results/17/full/predictions.jsonl#L27)

- **idx 245，详细描述与概括。** 两句都说最近的现生近亲分布在印度洋的塞舌尔。一句给出蛙科名称并说它们仅见于当地，另一句用马达加斯加附近来描述地理位置。金标为 1。BitFit 给出 0.294，完整微调给出 0.743。这个边界在于是否允许一对释义句保留不同详略。[BitFit 输出第 26 行](../results/17/bitfit/predictions.jsonl#L26) · [完整微调](../results/17/full/predictions.jsonl#L26)

种子 17 的仅分类头模型把 408 例中的 407 例判成正类，129 个负例全部判错。该次 F1 仍达 81.05，因为正类有 279 例；同时看 Accuracy 和假阳性数才能看出这个基线的局限。

## 单因素实验与解释边界

三项必做对照之后的单因素实验为 query 消融，已按相同种子执行。只把预训练部分的可训练集合从全部偏置缩为 12 个 query 偏置张量，其余设置与 BitFit 一致。这一对照测量移除其余偏置的合计影响，无法拆分某一层或某一类被移除偏置的贡献。

全部正式结果来自 RTX 4060 Laptop。昇腾的框架适配途径已查阅，本项目的 NPU 镜像和训练算子仍待实机核验。

[逐次指标及混淆矩阵 CSV](../results/summary.csv) · [独立复核脚本](../verify.py)
