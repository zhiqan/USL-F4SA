
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
#from umap import UMAP  # noqa


def visualize_memory(memory_bank: nn.Module,
                     save_path: str,
                     origin: str,
                     n_class: int = 25,
                     n_samples: int = 30,
                     proj: str = "tsne",
                     epoch: int = 0):
    """
    Visualizes the current state of the memory within DyCE

    Arguments:
        - memory_bank (nn.Module): DyCE memory module
        - save_path (str): path for saving plots
        - origin (str): student or teacher network
        - n_class (int): number of classes to visualize (optional)
        - n_samples (int): number of samples per class to visualize (optional)
        - proj (str): projection method (umap or tsne - optional)
        - epoch (int): epoch of the current memory state
    """
    print("==> Visualizing Memory Embeddings...")
    print(memory_bank.labels.unique(return_counts=True)[-1])
    unique_label_counts = np.array(
        memory_bank.labels.unique(return_counts=True)[-1].cpu())
    unique_labels = np.array(
        memory_bank.labels.unique(return_counts=True)[0].cpu())
    if len(unique_label_counts) <= 1:
        print("Error with memory configuration")
        return

    bank = np.array(memory_bank.bank.T.detach().cpu())
    labels = np.array(memory_bank.labels.detach().cpu())

    df_memory_bank = pd.DataFrame(bank)
    df_memory_bank['class'] = pd.Series(labels)
    df_memory_bank.to_csv('df_memory_bank.csv', index=False) 
    

    # discard classes with fewer than n_samples assgined to them
    discarded_labels = unique_labels[[idx for idx, val in enumerate(
        unique_label_counts) if val <= n_samples]]

    discarded_indices = np.argwhere(np.isin(unique_labels, discarded_labels))
    unique_labels = np.delete(unique_labels, discarded_indices)

    if len(unique_labels) < n_class:
        print("Error with memory configuration")
        return

    # randomly select n_class classes from the memory
    unique_labels = np.random.choice(unique_labels, n_class, replace=False)
    print("Unique after Sampling: {}".format(np.sort(unique_labels)))

    # keep only samples from selected claasses
    df_memory_bank = df_memory_bank[~df_memory_bank['class'].isin(
        discarded_labels)]
    # keep only n_samples per class for visualization purposes
    df_memory_bank = df_memory_bank[df_memory_bank['class'].isin(
        unique_labels)]

    # keep only n_samples per class for visualization purposes
    print("Df Unique classes: {}".format(
        np.sort(df_memory_bank['class'].unique())))

    try:
        df_memory_bank = df_memory_bank.groupby(
            'class', group_keys=False).apply(lambda df: df.sample(n_samples))
    except:
        print("Error with memory configuration")
        return
    # reset index
    df_memory_bank = df_memory_bank.reset_index(drop=True)

    # keep only features for projection
    features = df_memory_bank.iloc[:, :-1]
    if proj == "tsne":
        tsne = TSNE(n_components=2, verbose=0, perplexity=5)
        proj_2d = tsne.fit_transform(features)
    else:
        umap = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap.fit_transform(features)

    print("DBI score: {}".format(
        davies_bouldin_score(proj_2d, df_memory_bank['class'])))

    sns.set_context("paper")
    sns.set_style("ticks")
    ax = sns.relplot(x=proj_2d[:, 0], y=proj_2d[:, 1],
                     hue=df_memory_bank['class'].astype(int), palette="Dark2",
                     style=df_memory_bank['class'].astype(int), s=240,
                     legend=False, facet_kws=dict(despine=False))
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])

    save_path = Path(save_path) / Path("memory_visualizations") / Path(origin)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path / Path("memory_state_" +
                str(epoch)+".jpeg"), dpi=300)
    return


@torch.no_grad()
def visualize_optimal_transport(orginal_prototypes: np.array,
                                first_pass_prototypes: np.array,
                                transported_prototypes: np.array,
                                z_query: np.array,
                                y_support: np.array,
                                y_query: np.array,
                                proj: str,
                                episode: int,
                                n_shot: int,
                                save_path: str,
                                n_way: int = 5):
    """
    Visualizes {original, transported after first pass, and transported after 
    last pass} protoypes along with query set embeddings (and their distributions).

    Arguments:
        - orginal_prototypes (np.array): array of original supported prototypes
        - first_pass_prototypes (np.array): array of transported support prototypes
            after first OpTA pass
        - transported_prototypes (np.array): array of transported support prototypes
            after last OpTA pass
        - z_query (np.array): array of query features
        - y_support (np.array): series of support labels
        - y_query (np.array): series of query labels
        - z_query (np.array): array of query features
        - proj (str): projection method (umap or tsne)
        - episode (int): index of episode to be visualized
        - n_shot (int): number of images per class in episode
        - save_path (str): path for saving plots
        - n_way (int): number of classes in episode
    """
    df_sup_before = pd.DataFrame(orginal_prototypes)
    df_sup_before['class'] = pd.Series(y_support)
    df_sup_before['set'] = pd.Series(np.ones(len(df_sup_before.index)))

    df_sup_afterfp = pd.DataFrame(first_pass_prototypes)
    df_sup_afterfp['class'] = pd.Series(y_support)
    df_sup_afterfp['set'] = pd.Series(np.ones(len(df_sup_afterfp.index)))

    df_sup_after = pd.DataFrame(transported_prototypes)
    df_sup_after['class'] = pd.Series(y_support)
    df_sup_after['set'] = pd.Series(np.ones(len(df_sup_after.index)))

    df_quer = pd.DataFrame(z_query)
    df_quer['class'] = pd.Series(y_query)
    df_quer['set'] = pd.Series(np.zeros(len(df_quer.index)))

    df = pd.concat([df_sup_before, df_quer, df_sup_afterfp, df_sup_after])
    df = df.reset_index(drop=True)
    features = df.iloc[:, :-2]
    df.to_csv('prototypes'+str(episode)+'.csv', index=False) 
    
    


    tsne = TSNE(n_components=2, verbose=0, perplexity=5)
    proj_2d = tsne.fit_transform(features)
    df_proj_2d = pd.DataFrame(proj_2d, columns=['Dimension 1', 'Dimension 2'])
    df_proj_2d.to_csv('proj_2d'+str(episode)+'.csv', index=False)  # 保存为CSV文件
   

    proj_2d_before = proj_2d[:-2*n_way, :]
    proj_2d_afterfp = proj_2d[n_way:-n_way, :]
    proj_2d_after = np.concatenate(
        [proj_2d[n_way:-2*n_way, :], proj_2d[-n_way:, :]])

    print("Set DBI score before: {}".format(
        davies_bouldin_score(proj_2d_before, df['set'][:-2*n_way])))
    print("Set DBI score after first pass: {}".format(
        davies_bouldin_score(proj_2d_afterfp, df['set'][n_way:-n_way])))
    print("Set DBI score after last pass: {}".format(
        davies_bouldin_score(proj_2d_after, pd.concat([df['set'][n_way:-2*n_way],
                                                       df['set'][-n_way:]]))))

    # proj_2d_query = proj_2d[n_way:-2*n_way, :]
    # proj_2d_sup_before = proj_2d[:n_way]
    # proj_2d_sup_afterfp = proj_2d[-2*n_way:-n_way]
    # proj_2d_sup_after = proj_2d[-n_way:]

    # sns.set_context("paper")
    # sns.set_style("white")
    # proj_2d_query_df = pd.DataFrame(proj_2d_query, columns=["x", "y"])
    # proj_2d_query_df['class'] = pd.Series(df['class'][n_way:-2*n_way].astype(int))
    # proj_2d_query_df['set'] = pd.Series(df['set'][n_way:-2*n_way].astype(int))
    # proj_2d_sup_before_df = pd.DataFrame( proj_2d_sup_before, columns=["x", "y"])
    # ax2 = sns.scatterplot(data=proj_2d_query, x='x',
    #                       y='y',
    #                       hue='class',
    #                       s=250, style=df['set'][n_way:-2*n_way].astype(int),
    #                       palette="Dark2", markers=["."], legend=False)
    # ax2 = sns.scatterplot(data=proj_2d_sup_before, x=proj_2d_sup_before[:, 0],
    #                       y=proj_2d_sup_before[:, 1],
    #                       hue=df['class'][:n_way].astype(int), s=440,
    #                       style=df['set'][:n_way].astype(int), palette="Dark2",
    #                       markers=["d"], legend=False)
    # ax2.set(yticklabels=[])
    # ax2.tick_params(left=False)
    # ax2.set(xticklabels=[])
    # ax2.tick_params(bottom=False)
    # plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
    #             "_"+str(n_shot)+"-shot_before.jpeg"), dpi=300)
    # plt.clf()

    # # visualize first pass
    # proj_2d_sup_afterfp_df = pd.DataFrame(proj_2d_sup_afterfp, columns=["x", "y"])
    # ax = sns.kdeplot(data=proj_2d_query, x='x',y='y',
    #                  hue='class',
    #                  palette="Pastel2", legend=False)
    # ax = sns.scatterplot(data=proj_2d_query_df,x="x", y="y",
    #                      hue='class', s=130,
    #                      style='set',
    #                      palette="Dark2", markers=["."], legend=False)
    
    # ax = sns.scatterplot(data=proj_2d_sup_afterfp_df,x="x", y="y",
    #                      hue=df['class'][-2*n_way:-n_way].astype(int), s=800,
    #                      style=df['set'][-2*n_way:-n_way].astype(int),
    #                      palette="Dark2", markers=["*"], legend=False)
    # ax.set(yticklabels=[])
    # ax.tick_params(left=False)
    # ax.set(xticklabels=[])
    # ax.tick_params(bottom=False)
    # plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
    #             "_"+str(n_shot)+"-shot_after_first_pass.jpeg"), dpi=300)
    # plt.clf()

    # # visualize last pass
    # proj_2d_sup_after_df = pd.DataFrame(proj_2d_sup_after, columns=["x", "y"])
    # ax1 = sns.kdeplot(data=proj_2d_query,x='x',y='y',
    #                   hue=df['class'][n_way:-2*n_way].astype(int),
    #                   palette="Pastel2", legend=False)
    # ax1 = sns.scatterplot(data=proj_2d_query_df,x="x", y="y",
    #                       hue=df['class'][n_way:-2*n_way].astype(int), s=250,
    #                       style=df['set'][n_way:-2*n_way].astype(int),
    #                       palette="Dark2", markers=["."], legend=False)
    # ax1 = sns.scatterplot(data=proj_2d_sup_after_df,x="x", y="y",
    #                       hue=df['class'][-n_way:].astype(int), s=440,
    #                       style=df['set'][-n_way:].astype(int), palette="Dark2",
    #                       markers=["d"], legend=False)

    # ax1.set(yticklabels=[])
    # ax1.tick_params(left=False)
    # ax1.set(xticklabels=[])
    # ax1.tick_params(bottom=False)
    # plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
    #             "_"+str(n_shot)+"-shot_after_last_pass.jpeg"), dpi=300)
    # plt.clf()

    return
