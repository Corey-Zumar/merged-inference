import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
with open('../experiments/pytorch_multimodel/results_naive.json') as naive_json:
    naive_results = json.load(naive_json)
with open('../experiments/pytorch_multimodel/results_fused.json') as fused_json:
    fused_results = json.load(fused_json)


# def get_latency_graph():
#     bp_elems = []
#     fused_latencies = fused_results['stats']['batch_total_latencies']
#     bp_elems += ['ModelFuse' for _ in range(len(fused_latencies))]
#     print(len(fused_latencies))
#     print(len(bp_elems))
#     naive_latencies = naive_results['stats']['batch_total_latencies']
#     bp_elems += ['Naive' for _ in range(len(naive_latencies))]
#     print(naive_latencies)
#     fused_mean = np.mean(fused_latencies)
#     naive_mean = np.mean(naive_latencies)
#     print(naive_mean)
#     print(fused_mean)
#     print(len(bp_elems))
#     print(len(fused_latencies + naive_latencies))
#     all_latencies = fused_latencies + naive_latencies
#     df = pd.DataFrame(dict(bp_elems=bp_elems, latency=all_latencies))
#     ax = sns.barplot(x="bp_elems", y="latency", data=df, ci="sd")
#     ax.set(xlabel="Latency (seconds)", ylabel="Serving Platform")
#     return ax

def get_attr_graph(attr, yAxisLabel):
    bp_elems = []
    fused_attrs = fused_results['stats'][attr]
    bp_elems += ['ModelFuse' for _ in range(len(fused_attrs))]
    naive_attrs = naive_results['stats'][attr]
    bp_elems += ['Naive' for _ in range(len(naive_attrs))]
    all_attrs = fused_attrs + naive_attrs
    df = pd.DataFrame(dict(bp_elems=bp_elems, attribute=all_attrs))
    ax = sns.barplot(x="bp_elems", y="attribute", data=df, ci="sd")
    ax.set(xlabel="Serving Platform", ylabel=yAxisLabel)
    return ax


def get_throughput_graph():
    return get_attr_graph('thrus', 'Throughput (images/second)')


def get_latency_graph():
    return get_attr_graph('batch_total_latencies', 'Latency (seconds)')


ax = get_latency_graph()
plt.show()


