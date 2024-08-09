import tensorboard as tb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 300
if __name__ == "__main__":
    results = pd.read_csv("results/hyperparams_cv.csv")
    results = results.set_index("Experiment ID",drop=True)
    results = results.T
    exp_names = results.index
    plt.figure(figsize=(10,5))
    # exp_names = [f"experiment {i}\n ({n})" for i,n in enumerate(results.index)]
    exp_names = [f"experiment {i}" for i in range(len(results.index))]
    fold_res = results[["fold1","fold2","fold3","fold4","fold5"]]
    results["mean"] = results["mean"].str.replace(",",".").astype('float').map(lambda x: f"{x:.3E}")
    results["standard deviation"] = results["standard deviation"].str.replace(",",".").astype('float').map(lambda x: f"{x:.3E}")
    results["lower bound"] = results["lower bound"].str.replace(",",".").astype('float').map(lambda x: f"{x:.3E}")
    results["upper bound"] = results["upper bound"].str.replace(",",".").astype('float').map(lambda x: f"{x:.3E}")

    means = results["mean"].values.astype("float")
    std = results["standard deviation"].values.astype("float")
    for i,exp_name in enumerate(exp_names):
        plt.scatter(x=[exp_name]*5,y=fold_res.iloc[i].str.replace(",",".").astype('float').values)
        plt.errorbar(x=exp_name, y=means[i], yerr=std[i], color='black', fmt='_', capsize=3)
    
    plt.ylabel("Mean square error of forecasted bins",fontsize=12)
    plt.savefig("results/hyperparams.jpg")
    plt.close()

    col_indices = [-9,-7,-6,-5,-4,-3,-2]
    hyparparams_dict = results[results.columns[col_indices]].to_dict()
    celltext = [list(v.values()) for (k,v) in hyparparams_dict.items()]
    fig, (ax, ax_table) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3,1]))
    for i,exp_name in enumerate(exp_names):
        ax.scatter(x=[exp_name]*5,y=fold_res.iloc[i].str.replace(",",".").astype('float').values)
        ax.errorbar(x=exp_name, y=means[i], yerr=std[i], color='black', fmt='_', capsize=3)

    ax.set_ylabel("Mean square error of forecasted bins",fontsize=8)

    ax_table.axis('off')
    ax.tick_params(axis='x', labelrotation=45,labelsize=8)
    ax_table = plt.table(cellText=celltext,
            rowLabels=results.columns[col_indices],
            bbox=[0, -0.5, 1, 1],
            loc='bottom',)
    ax_table.auto_set_font_size(False)
    ax_table.set_fontsize(6)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.2, bottom=0.2)
    plt.savefig("results/hyperparams2.jpg",)
    plt.close()