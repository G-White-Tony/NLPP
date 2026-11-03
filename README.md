# BitFit 在 MRPC 上的课程实验

[中期进展、结果与误判分析](docs/midterm.md)

## 固定资源与实验设置

- 数据为 GLUE 的 MRPC，使用官方全部 3,668 条 train、408 条 validation。预检各取前 16 条。正式实验无抽样，原始 idx 和顺序见 [训练 ID](data/train_ids.json) 与 [验证 ID](data/validation_ids.json)。标签 1 表示释义。全程只用训练集更新权重。数据与模型于 2026-11-03 获取。[数据版本](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/mrpc)
- 模型为 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)。仅分类头重新初始化。数据和分词器等通过 HF 镜像获取，权重通过 ModelScope 同名模型获取，并与固定 HF 版本的 LFS SHA-256 核对。[文件哈希](data/resources.json)
- 对照参照[作者实现](https://github.com/benzakenelad/BitFit/tree/7ead19a8350a01d5701f9e2df896a1c5b42c3723)。本组独立实现训练与审计。full 开放所有参数，head 仅开放分类头，bitfit 开放分类头和 BERT 所有 bias，query 仅开放分类头和各层 query.bias。query 冻结 pooler bias，与作者代码在部分 bias 实验中额外开放 pooler bias 有差别。
- 种子 17、42、73 分别控制分类头初始化、dropout 与逐轮乱序。同种子复用相同初始化与数据顺序。每轮覆盖整份训练集，最后不足 8 条的批次保留。固定取第 3 轮末模型评价，不按验证分数挑轮次。
- batch 8，最大长度 128，拼接句对，超长按 longest-first 截断。BF16 自动混合精度、FP32 参数、AdamW、初始学习率 full 为 2e-5，其余为 1e-3，线性衰减至 0，无 warmup，weight decay 0，梯度范数上限 1。完整微调和 BitFit 的比较包含各自预设学习率的影响；BitFit 对 query 的消融只改可更新参数。
- 指标为正类 F1 和 Accuracy。训练耗时包含三个 epoch 和首步梯度审计，起止均同步 CUDA。显存是训练过程 PyTorch 峰值 allocated MiB，含常驻 GPU 的分词后数据。训练后的参数差值审计、验证、下载和写盘均在计时区间外。
- Windows、RTX 4060 Laptop 8 GB、驱动 581.57、Python 3.10.20、PyTorch 2.1.2+cu121、Transformers 4.37.2。资源 G 本次实际使用 CUDA。昇腾官方提供 PyTorch 适配插件；具体 NPU、CANN、torch_npu 镜像组合与本任务算子仍待上机验证。[昇腾官方适配说明](https://github.com/Ascend/pytorch)

## 复现

在仓库根目录和独立 Python 3.10 环境执行。下载受限时可设置 HF_ENDPOINT 指向可用的 Hugging Face 镜像，或将相同哈希的文件放入 .cache 对应位置。prepare.py 会校验缓存并生成训练输入。

```powershell
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers==4.37.2 numpy==1.26.4 tokenizers==0.15.1 safetensors==0.7.0 pyarrow==23.0.0
python -s prepare.py
python -s train.py --smoke
python -s train.py
```

单独复核某组可执行 `python -s train.py --mode bitfit --seed 42`。相同配置会覆盖该次结果。results 下按种子和方法保存每轮损失、指标、逐条预测与逐张量审计。predictions.jsonl 的 idx 对应固定验证 ID，p1 是正类概率。audit.json 记录首步梯度范数、初始到训练结束发生变化的元素数和最大绝对差。冻结参数应无梯度、差值为零。

全量运行后执行 `python -s verify.py`，脚本会用原始验证标签重新计算 12 次 F1、Accuracy、混淆矩阵，并检查同种子初始分类头哈希和各参数张量的梯度、实际更新范围。
