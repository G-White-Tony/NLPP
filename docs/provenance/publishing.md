# 本机实验与仓库文件映射

根目录 train.py、verify.py、results/ 和 docs/midterm.md 保留远端原始12次实验。旧README存档于 docs/README_original.md。

本机第一轮实际执行的 train.py 原字节内容发布为根目录 train_local.py，其SHA-256与 results_local/environment.json 中的 training_script_sha256 一致。此重命名是为了不覆盖原仓库脚本，不改变训练实现。已执行的调度脚本原件保留于 executed/；发布版 run_suite.py 的调用路径、run_data_suite.py 的基线代码哈希读取路径改为 train_local.py，运行逻辑不变。train_data.py、train_layers.py及原始实验JSON和日志按字节复制。

本机运行记录：results_local 21次、results_data 18次、results_layers 9次，共48次；原 results 的12次属于另一软件环境，单独保留，不混算统计量。预检 smoke 与单批次graph_probe不计入正式训练。

仓库仅发布可复现源代码、固定输入版本与哈希、子集ID、预测、审计和指标，不包含.cache中的预训练权重和临时运行环境。原始验证样例来自固定版本公开MRPC，examples/包含输入、标签和预测来源。报告Markdown只调整图片相对链接以适配GitHub；PDF和Word为此前生成的最新报告原件。
