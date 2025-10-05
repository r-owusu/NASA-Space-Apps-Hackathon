import os
import pandas as pd
import matplotlib.pyplot as plt
def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
def plot_histograms(dfs, labels, column, out_path, title):
    plt.figure(figsize=(10,4))
    for df,label in zip(dfs,labels):
        plt.hist(df[column], bins=50, alpha=0.5, label=label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
