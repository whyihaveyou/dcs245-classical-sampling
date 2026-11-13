"""Render verified fullscore results; requires successful analyze.py output."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent;out=base/'analysis';s=json.loads((out/'summary.json').read_text());assert s['validation_passed']
rows=[json.loads(l) for l in (out/'scores.jsonl').read_text().splitlines()]
ibm=list(csv.DictReader((base.parent/'protocol/zenodo_22882048/zenodo_experiment1_data/experiment1_probabilities.csv').open()))
x=np.array([2**64*r['probability_fp64'] for r in rows]);y=np.array([2**64*float(r['p']) for r in ibm])
a=s['classical_r1'];b=s['IBM_hardware'];d=s['difference']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':11,'xtick.direction':'in','ytick.direction':'in','axes.spines.top':True,'axes.spines.right':True})
fig,axs=plt.subplots(1,2,figsize=(8.4,3.7),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
bins=np.linspace(0,max(x.max(),y.max())*1.001,36)
colors=['#1764ab','#c65f19'];labels=['Classical r1','IBM hardware']
for z,c,label in zip([x,y],colors,labels):axs[0].hist(z,bins=bins,density=True,histtype='step',linewidth=1.6,color=c,label=label)
axs[0].set_xlabel(r'$z=2^{64}p(x)$');axs[0].set_ylabel(r'$\rho(z)$');axs[0].legend(frameon=False,loc='upper right');axs[0].set_ylim(0,axs[0].get_ylim()[1]*1.15)
for i,v in enumerate([a,b]):
 ci=v['bootstrap95_percentile_iid'];m=v['linear_xeb'];axs[1].errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='o',color=colors[i],capsize=5,markersize=6)
axs[1].set_xticks([0,1],labels);axs[1].set_xlim(-.6,1.6);axs[1].set_ylim(0,max(a['bootstrap95_percentile_iid'][1],b['bootstrap95_percentile_iid'][1])*1.22);axs[1].set_ylabel(r'$\widehat{\mathrm{XEB}}$')
for ax,letter in zip(axs,['a','b']):ax.text(.04,.91,f'({letter})',transform=ax.transAxes,va='top')
fig.savefig(out/'dcs245_fullscore.png',dpi=200);fig.savefig(out/'dcs245_fullscore.pdf');plt.close(fig)
caption='Full probability scoring of QAT #245 samples. (a) Empirical densities of scaled reference probabilities z = 2^64 p(x) for 1,389 classical r1 outputs and 1,389 IBM hardware outputs. All samples are included. Classical probabilities use a fresh complex128 contraction; IBM probabilities are from the independently published #249 data. (b) Linear XEB means with 95% percentile bootstrap intervals, 20,000 sample-level resamples. Hardware temporal correlations are not modeled. Linear XEB is not a fidelity certificate.'
(out/'figure_caption.txt').write_text(caption+'\n')
fmt=lambda v:f"{v['linear_xeb']:.6f} ± {v['standard_error_iid']:.6f}"
ci=lambda v:'['+', '.join(f'{t:.6f}' for t in v['bootstrap95_percentile_iid'])+']'
worker_table='\n'.join(f"| {w['worker_id']} | {w['sample_count']} | {w['score_seconds']:.3f} |" for w in s['workers'])
ledger=s['central_timing'];report=f'''# QAT 245 全量 complex128 打分结果

r1 的全部 **1,389 条经典样本**已完成完整缩并的 complex128 概率重算。Linear XEB 为 **{fmt(a)}**，95% bootstrap 区间为 **{ci(a)}**；IBM 的 1,389 条硬件样本为 **{fmt(b)}**。全部记录和已有 64 条 FP64 审计锚点通过核验。

## 结果及比较

| 数据 | 样本数 | Linear XEB ± 标准误 | 95% bootstrap 区间 |
|---|---:|---:|---:|
| 经典 r1 | 1,389 | {fmt(a)} | {ci(a)} |
| IBM 硬件 | 1,389 | {fmt(b)} | {ci(b)} |

两者 XEB 差值为 {d['linear_xeb']:.6f}，独立样本标准误 {d['standard_error_independent_iid']:.6f}；差值 bootstrap 区间为 {d['bootstrap95_independent_iid']}。这里的 XEB 是分布打分，不等同于保真度。原有 64 条预选审计的保真度下界是另一项结果。

![全量概率与 XEB 比较]({out/'dcs245_fullscore.png'})

## 数据和计算方法

经典输入直接来自冻结 r1 的八份 `samples.jsonl`，按 sample_id 排序保留全部记录及重复值；不是旧脚本默认读取的 IBM 样本。完整样本序号为 0–1388。样本列表 SHA256 为 `{s['hashes']['r1_sample_list']}`。

使用 r1 原有图、位序、自由顶点和 `inplace_k3_fp64` 计划，重新计算每个 44 位前缀对应的完整 2^20 维 complex128 条件向量。由对应 A-index 取振幅 v，计算 p(x)=|v|²/2^44，再以 mean(2^64 p(x))−1 求 XEB。所有切片完整求和，无张量截断；complex128 仍是浮点数值参考，不是区间算术证明。此重算独立于原 FP32 生成过程，但沿用同一图和核心缩并算法，不能称为完全独立的另一种算法。

IBM 对照使用 [QAT 249](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/249) 所公开的 [Zenodo 概率数据](https://zenodo.org/records/22882048)，在本地重新计算统计量。其 CSV SHA256 为 `{s['hashes']['ibm_csv']}`。

## 独立收回核验

- 8 个单卡作业全部成功，计数 174/174/174/174/174/173/173/173，总计 1,389，无缺失、无重复 sample_id。
- 样本源文件哈希、冻结源码哈希、Qiskit 位序到逻辑位序映射、A-index、振幅到概率和 XEB 公式全部检查通过。
- 每条 stdout checkpoint 与最终压缩归档逐项一致。
- 64 条历史 FP64 锚点全部复核，概率最大相对差为 {s['max_anchor_relative_probability_error']:.3g}。
- 全部抽中点的原 FP32 概率估计对新 FP64 分数的最大相对差为 {s['selected_point_FP32_vs_FP64']['max_relative_error']:.3g}；相对差大于 1e-4 的点数为 {s['selected_point_FP32_vs_FP64']['count_above_1e_4']}。这是抽中点检查，不是全分布 TV 证书。
- 统计采用 ddof=1 标准误、固定 seed=20261113 的 20,000 次 percentile bootstrap。上述区间采用样本独立假设，未校正 IBM 硬件的时间相关性。

## 评分资源和时间

本次为 **8 个独立单卡 H200 作业**，每卡主缓冲区 128 GiB。下列均为补充验证成本，不是生成 1,389 条样本的时间，不能拿它替代既有生成计时，也不能据此宣称端到端超过 #245 的 870 秒。

| 分片 | 概率数量 | 评分进程时间 秒 |
|---|---:|---:|
{worker_table}

最长单卡评分进程 {s['max_worker_scoring_seconds']:.3f} 秒；累计评分进程占卡时间 {s['GPU_hours_scoring_process_sum']:.4f} GPU·小时。中央连续 monotonic 从首个提交开始到全部日志收回开发机为 **{ledger['central_monotonic_submit_to_all_logs_seconds']:.3f} 秒**，包含提交、排队、评分和日志回收；不包含此前单卡验证、源文件准备，也不包含随后本地下载、CPU 核验和绘图。GPU·小时不含容器启动和提取依赖的开销。

## 可复现文件

- [统计和核验摘要]({out/'summary.json'})
- [全量概率 CSV]({out/'scores.csv'})
- [完整逐样本结果]({out/'scores.jsonl'})
- [独立 CPU verifier 和统计脚本]({base/'analyze.py'})
- [绘图脚本]({base/'make_report.py'})
- [冻结评分 manifest]({base/'source/scoring_manifest.json'})
- [中央连续计时]({base/'production/central_timing.json'})
- [论文用矢量图]({out/'dcs245_fullscore.pdf'})

这些结果补齐论文和 #263 所需的全量 XEB 证据。本轮只生成本地归档，未对外发布。
'''
report_path=base.parents[2]/'outputs/dcs245_fullscore_results_2026-11-13.md'
report_path.write_text(report)
print(report_path)
