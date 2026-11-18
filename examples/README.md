# 逐样本输入与模型输出

样例来自固定版本 GLUE MRPC 验证集。标签1表示释义，0表示非释义；全部比较固定种子17，p1为正类概率。先固定追踪报告中的10例，再按验证集原顺序各取3例“顶部误判而交错正确”和“交错误判而底部正确”。这种选择用于诊断，不代表随机抽样错误率；包括反例以避免只展示改进。

机器可读记录见 [cases.jsonl](cases.jsonl)。全部408条预测见各实验结果目录。

## idx 284　固定历史错误与边界例追踪

句子一：Cooley said he expects Muhammad will similarly be called as a witness at a pretrial hearing for Malvo .

句子二：Lee Boyd Malvo will be called as a witness Wednesday in a pretrial hearing for fellow sniper suspect John Allen Muhammad .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9746 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7245 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.5407 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7953 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7533 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.4574 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.8515 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.3777 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 245　固定历史错误与边界例追踪

句子一：Its closest living relatives are a family frogs called sooglossidae that are found only in the Seychelles in the Indian Ocean .

句子二：Its closest relative is found in the Seychelles Archipelago , near Madagascar in the Indian Ocean .

真实标签：**1**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.1795 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.5583 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.1999 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 0 | 0.2494 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 0 | 0.3541 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.2975 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 0 | 0.2576 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.2830 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 54　固定历史错误与边界例追踪

句子一：While dioxin levels in the environment were up last year , they have dropped by 75 percent since the 1970s , said Caswell .

句子二：The Institute said dioxin levels in the environment have fallen by as much as 76 percent since the 1970s .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9976 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6909 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.8069 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.6907 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7004 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.7652 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.9329 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.9019 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 127　固定历史错误与边界例追踪

句子一：Unable to find a home for him , a judge told mental health authorities they needed to find supervised housing and treatment for DeVries somewhere in California .

句子二：The judge had told the state Department of Mental Health to find supervised housing and treatment for DeVries somewhere in California .

真实标签：**1**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0181 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7169 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.3012 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.8022 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.8559 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.9386 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.9284 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.7592 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 222　固定历史错误与边界例追踪

句子一：The driver , Eugene Rogers , helped to remove children from the bus , Wood said .

句子二：At the accident scene , the driver was " covered in blood " but helped to remove children , Wood said .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0109 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6759 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.6145 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7404 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7976 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.4685 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.8856 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.6743 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 334　固定历史错误与边界例追踪

句子一：With these assets , Funny Cide has a solid chance to become the first Triple Crown winner since Affirmed in 1978 .

句子二：Funny Cide is looking to become horse racing 's first Triple Crown winner in a generation .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.1259 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6632 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.8256 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.6347 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 0 | 0.3518 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.4499 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 0 | 0.2720 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.2740 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 374　固定历史错误与边界例追踪

句子一：Bush wanted " to see an aircraft landing the same way that the pilots saw an aircraft landing , " White House press secretary Ari Fleischer said yesterday .

句子二：On Tuesday , before Byrd 's speech , Fleischer said Bush wanted ' ' to see an aircraft landing the same way that the pilots saw an aircraft landing .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9978 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7169 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.9787 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.8561 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.9250 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.7759 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.9696 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.9875 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 548　固定历史错误与边界例追踪

句子一：In the 2002 study , the margin of error ranged from 1.8 to 4.4 percentage points .

句子二：It has a margin of error of plus or minus three to four percentage points .

真实标签：**1**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0260 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6605 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.2095 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 0 | 0.3306 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 0 | 0.2313 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.2491 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 0 | 0.3650 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.1582 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 594　固定历史错误与边界例追踪

句子一：Secretary of State Colin Powell designated the Chechen leader believed responsible for last year 's hostage standoff in a Moscow theater as a threat to U.S. security Friday .

句子二：U.S. Secretary of State Colin Powell on Friday designated Chechen rebel leader Shamil Basayev a threat to the security of the United States and to U.S. citizens .

真实标签：**1**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9872 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7232 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.3632 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7684 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7656 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.2302 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.8563 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.6448 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 694　固定历史错误与边界例追踪

句子一：McCabe said he was considered a witness , not a suspect .

句子二：" He is not considered a suspect , " McCabe said .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9956 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6252 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.9810 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 0 | 0.3755 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 0 | 0.4707 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.9511 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.7443 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.9247 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 25　顶部误判而交错正确

句子一：The dollar was at 116.92 yen against the yen , flat on the session , and at 1.2891 against the Swiss franc , also flat .

句子二：The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0072 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6971 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.0940 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7139 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.5706 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.1242 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.5998 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.0927 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 33　顶部误判而交错正确

句子一：No dates have been set for the civil or the criminal trial .

句子二：No dates have been set for the criminal or civil cases , but Shanley has pleaded not guilty .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0072 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.6918 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.1727 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.6700 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7786 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.5648 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.7186 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.4059 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 83　顶部误判而交错正确

句子一：" Sanitation is poor ... there could be typhoid and cholera , " he said .

句子二：" Sanitation is poor , drinking water is generally left behind . . . there could be typhoid and cholera . "

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0068 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7140 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 0 | 0.0981 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.8245 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.9281 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.8670 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.8578 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.3950 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 718　交错误判而底部正确

句子一：On Saturday , a 149mph serve against Agassi equalled Rusedski 's world record .

句子二：On Saturday , Roddick equalled the world record with a 149 m.p.h. serve in beating Andre Agassi .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9615 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7000 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.7925 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.6365 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.5227 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.3749 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.7364 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.6221 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 1076　交错误判而底部正确

句子一：According to the Merchant Marine Ministry , the 37-year-old ship is registered to Alpha Shipping Inc. based in the Pacific Ocean nation of Marshall Islands .

句子二：The Baltic Sky is a 37-year-old ship registered to Alpha Shipping Inc. based in the Pacific Ocean nation of Marshall Islands .

真实标签：**1**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 1 | 0.9971 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7003 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.7950 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7179 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.6111 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 1 | 0.9466 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.9143 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 0 | 0.4717 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |

## idx 1470　交错误判而底部正确

句子一：President Bush raised a record-breaking $ 49.5 million for his re-election campaign over the last three months , with contributions from 262,000 Americans , the president 's campaign chairman said Tuesday .

句子二：President Bush has raised $ 83.9 million since beginning his re-election campaign in May , and has $ 70 million of that left to spend , his campaign said Tuesday .

真实标签：**0**

| 方法 | 预测 | p1 | 证据 |
|---|---:|---:|---|
| full | 0 | 0.0176 | [原始JSONL](../results_local/17/full_lr2e-05/predictions.jsonl) |
| head | 1 | 0.7081 | [原始JSONL](../results_local/17/head_lr1e-03/predictions.jsonl) |
| bitfit | 1 | 0.5266 | [原始JSONL](../results_local/17/bitfit_lr1e-03/predictions.jsonl) |
| query | 1 | 0.7377 | [原始JSONL](../results_local/17/query_lr1e-03/predictions.jsonl) |
| query_intermediate | 1 | 0.7635 | [原始JSONL](../results_local/17/query_intermediate_lr1e-03/predictions.jsonl) |
| bottom | 0 | 0.3474 | [原始JSONL](../results_layers/17/bottom_lr1e-03/predictions.jsonl) |
| top | 1 | 0.7287 | [原始JSONL](../results_layers/17/top_lr1e-03/predictions.jsonl) |
| alternating | 1 | 0.6560 | [原始JSONL](../results_layers/17/alternating_lr1e-03/predictions.jsonl) |
