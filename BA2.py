import torch
import torch.nn.functional as F
from lightly.loss.memory_bank import MemoryBankModule
from sklearn import cluster
from torchmetrics.functional import pairwise_euclidean_distance

from utils.sinkhorn import distributed_sinkhorn
from utils.visualize import visualize_memory
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.to(device)
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components





class BA(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16, origin: str = None):
     
        super(BA, self).__init__(size)
        # register buffers for stored prototypes and labels in the memory
        self.register_buffer(
            "prototypes", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "labels", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )
        
        self.register_buffer("indexs", torch.randn(size), persistent=False)
        self.register_buffer(
            "index_bank", torch.full((size,), -1, dtype=torch.long), persistent=False
        )

        self.start_clustering = False
        self.last_cluster_epoch = 1
        self.last_vis_epoch = 1
        self.origin = origin
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)
        
        

    @torch.no_grad()
    def cluster_memory_embeddings(self,
                                  cluster_algo: str = "kmeans",
                                  num_clusters: int = 300,
                                  min_cluster_size: int = 4):
       
        bank = self.bank.detach()
        #print(bank.shape)

        bank_np = F.normalize(bank.detach()).cpu().numpy()

        if cluster_algo == "kmeans":
            clf, labels, _ = clusterer(bank_np,
                                       n_clusters=num_clusters,
                                       algo="kmeans")
            # get cluster means & labels
            prototypes = clf.cluster_centers_
            prototypes = torch.from_numpy(prototypes).type_as(bank).cpu()
            labels = torch.from_numpy(labels).type_as(bank).long().cpu()
            # do not upddate the clusters in the memory in case of redundancy
            # from kmeans
            if self.start_clustering and len(labels.unique(return_counts=True)[-1]) < num_clusters:
                return
            self.prototypes = prototypes
            self.labels = labels

    @torch.no_grad()
    def save_memory_embeddings(self,
                               args: dict,
                               z: torch.Tensor,
                               index: torch.Tensor, 
                               dist_metric: str = "euclidean",
                               momentum=0.9):
        """
        Finds the optimal assignments between current batch embeddings and stored
        memory prototypes and ensures equipartitioning of assignments. The optimal
        assignments are then used for updating the memory protoypes and  
        partition.

        Arguments:
            - args (dict): parsed keyword training arguments: 
            - z (torch.Tensor): the input batch embeddings
            - dist_metric (str): choice of distance metric for calculating 
                distance matrix for optimal transport (optional)
            - momentum (float): momentum parameter for updating the memory 
                prototypes (optional)
        """
        prototypes = self.prototypes.clone().cpu()
        prototypes = prototypes - prototypes.mean(dim=0)
        #prototypes1=PCA_svd(prototypes, 512, center=True)
        
        
        # # 2. 计算协方差矩阵 (2049x2049)
        # cov_matrix = torch.mm(data_centered.T, data_centered) / (data_centered.size(0) - 1)
        
        # # 3. 计算协方差矩阵的特征值和特征向量
        # eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        # # eigenvectors (2049x2049), 代表个特征的主成分方向
        
        # # 4. 选择前 k 个主成分
        # k = 100  # 降维目标，比如降到 10 维
        # _, indices = torch.sort(eigenvalues[:, 0], descending=True)  # 按特征值排序
        # principal_components = eigenvectors[:, indices[:k]]  # 选择前 k 个主成分
        
        # # 5. 将数据投影到低维空间
        # prototypes = torch.mm(data_centered, principal_components)  # (100, 10)
        #print(prototypes1.shape)

        if dist_metric == "cosine":
            # Normalize batch & memory embeddings
            z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
            prototypes_normed = torch.nn.functional.normalize(
                prototypes, dim=1).to(device)  # K x D

            # create cost matrix between batch embeddings & cluster prototypes
            Q = torch.einsum("nd,md->nm", z_normed,
                             prototypes_normed)  # BS x K
        else:
            if args.eucl_norm:
                # Normalize batch & memory embeddings
                z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
                prototypes_normed = torch.nn.functional.normalize(
                    prototypes, dim=1).to(device) # K x D
                
                # create cost matrix between batch embeddings & cluster prototypes
                Q = pairwise_euclidean_distance(z_normed, prototypes_normed)
                has_nan_or_infa = torch.isnan(Q).any() or torch.isinf(Q).any()
                #print("Q中NaN or Inf values:", has_nan_or_infa)
                # if has_nan_or_infa:
                    
                #     sys.exit()
            else:
                # create cost matrix between batch embeddings & cluster prototypes
                Q = pairwise_euclidean_distance(z, prototypes.to(device))

        # apply optimal transport between batch embeddings and cluster prototypes
        Q = distributed_sinkhorn(
            Q, args.epsilon, args.sinkhorn_iterations)  # BS x K

        # get assignments (batch labels)
        batch_labels = torch.argmax(Q, dim=1)

        # add equipartitioned batch to memory and discard oldest embeddings
        batch_size = z.shape[0]
        ptr = int(self.bank_ptr)
        #print(ptr)
       # print(self.size)
        #print(self.size[0] - ptr)
        #print('bankshape',self.bank.shape)
        #print('zshape',z.shape)
        
        if ptr + batch_size >= self.size[0]:
        
            self.bank[ptr:, ] = z[: self.size[0] - ptr].detach()
            self.labels[ptr:] = batch_labels[: self.size[0] - ptr].detach()
            self.index_bank[ptr:] = index[: self.size[0] - ptr].detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[ptr: ptr + batch_size,:] = z.detach()
            self.labels[ptr: ptr + batch_size] = batch_labels.detach()
            self.index_bank[ptr: ptr + batch_size] = index.detach()
            self.bank_ptr[0] = ptr + batch_size

        # Update cluster prototypes
        labels = self.labels.clone().cpu()
        bank = self.bank.clone().cpu().detach()
        index_bank = self.index_bank.clone().cpu().detach()
        
        #print('bankshape_af',self.bank.shape)
        #print('labels',labels.shape)

        view = labels.view(labels.size(0), 1).expand(-1, bank.size(1))
        unique_labels, labels_count = view.unique(dim=0, return_counts=True)
        deleted_labels = []
        for i in range(0, prototypes.shape[0]):
            if i not in unique_labels[:, 0]:
                deleted_labels.append(i)
                label = torch.tensor([[i]]).expand(-1, bank.size(1))
                unique_labels = torch.cat((unique_labels, label), 0)
                labels_count = torch.cat(
                    (labels_count, torch.tensor([0.001])), 0)

        # get cluster means
        #print('delabel',deleted_labels)
        #print('unique_labels',unique_labels.shape)
        prototypes_next = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
            0, view.type(torch.int64), bank).cpu()  # UN x 512
        prototypes_next = prototypes_next / labels_count.float().unsqueeze(1)
        #print("prototypes shape:", prototypes.shape)
        #print("prototypes_next shape:", prototypes_next.shape)

        # in case a stored cluster ends up without any assignments, use the
        # previous value of its prototype as the new prototype
        for i in deleted_labels:
            prototypes_next[i, :] = prototypes[i, :prototypes_next.shape[1]]

        # EMA update of cluster prototypes
        self.prototypes = prototypes_next * \
            (1 - momentum) + momentum * prototypes

        return z, bank,index_bank,labels
    def get_bank_indices(self):
        return self.index_bank.clone()


    def get_top_kNN(self,
                    output: torch.Tensor,
                    bindexs: torch.Tensor,
                    epoch: int,
                    args,
                    k: int = 5):
        # pointer of memory bank (to know when it fills for the first time)
        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0
        bsz = output.shape[0]
        #self.indexs=bindexs
        #print('ptr',ptr)
        #print('bank_ptr',self.bank_ptr)
        # current_ptr = self.bank_ptr1.item()
        # self.bank_ptr1[0]=(current_ptr+bsz)%self.size[0]
        np.save('output_chushi.npy', output.detach().numpy())
        np.save('bankindexs_chushi.npy', bindexs.detach().numpy())

        if self.start_clustering == False:
            # if clusters not yet initialized
            if ptr + bsz >= self.size[0]:
                # if memory is full for the first time
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)

                self.last_cluster_epoch = epoch
                self.indexs[ptr:] = bindexs[: self.size[0] - ptr].detach()
                self.start_clustering = True

                # Add latest batch to the memory queue using Optimal Transport
                output, bank,index_bank,labels = self.save_memory_embeddings(
                    args, output,bindexs, dist_metric=args.memory_dist_metric,
                    momentum=args.memory_momentum)
                prototypes_initialized = True
                np.save('optbank1.npy', bank.detach().numpy())
                np.save('optbankindexs1.npy', index_bank.detach().numpy())
                np.save('label1.npy', labels.detach().numpy())
            else:
                #print(self.bank)
                # Add latest batch to the memory (memory not yet full for first time)
                output, bank = super(BA, self).forward(
                    output, None, update=True)
                prototypes_initialized = False
                self.indexs[ptr: ptr + bsz] = bindexs.detach()
                self.index_bank[ptr : ptr + bsz] = bindexs.detach()
                np.save('bank.npy', bank.detach().numpy())
                np.save('bankindexs.npy', self.index_bank.detach().numpy())
                #print('index',self.indexs.shape)
                #print('bindex',bindexs.shape)
                
        else:
            # cluster are now initialized
            if args.recluster and epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
                # periodically restart the memory clusters if reclustering is enabled
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo,
                    num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            if len(self.labels.unique(return_counts=True)[-1]) <= 1:
                # restart memory clusters in case the memory embeddings have
                # converged to a single cluster
                # (In practice: not used, but covers the case when not suitable
                # hyperparameters for the OT memory updating have been chosen)
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo,
                    num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            # visualize memory embeddings using UMAP with an args.visual_freq
            # epoch frequency
            if epoch % args.visual_freq == 0 and epoch != self.last_vis_epoch:
                self.last_vis_epoch = epoch
                if self.origin in ["teacher", "student"]:
                    visualize_memory(self, args.save_path,
                                     self.origin, epoch=epoch,
                                     n_samples=args.memory_scale)

            # Add latest batch to the memory queue using Optimal Transport
            output, bank,index_bank,labels = self.save_memory_embeddings(
                args, output,bindexs,dist_metric=args.memory_dist_metric, momentum=args.memory_momentum)
            prototypes_initialized = True
            np.save('optbank2.npy', bank.detach().numpy())
            np.save('optbankindexs2.npy', index_bank.detach().numpy())
            np.save('label2.npy', labels.detach().numpy())
            if ptr + bsz >= self.size[0]:
                self.indexs[ptr:] = bindexs[: self.size[0] - ptr].detach()
            else:
                self.indexs[ptr: ptr + bsz] = bindexs.detach()
                
                

        bank = bank.to(output.device).t()
        # only concat the nearest neighbor features in case the memory start
        # epoch has passed (i.e. the memory has converged to stable state)
        if epoch >= args.memory_start_epoch:
            # Normalize batch & memory embeddings
            output_normed = torch.nn.functional.normalize(output, dim=1)
            bank_normed = torch.nn.functional.normalize(bank.T, dim=1)
            
            #print(bank_normed.shape)
            

            # split embeddings of the 2 views
            z1, z2 = torch.split(
                output_normed, [args.batch_size, args.batch_size], dim=0)
            indexs1,indexs2=torch.split(
                bindexs, [args.batch_size, args.batch_size], dim=0)
            #print('z1_shape',z1.shape)

            # create similarity matrix between batch & memory embeddings
            similarity_matrix1 = torch.einsum(
                "nd,md->nm", z1, bank_normed)
            similarity_matrix2 = torch.einsum(
                "nd,md->nm", z2, bank_normed)

            # if the stored prototypes are initialized
            if prototypes_initialized:
                prototypes = self.prototypes.clone().to(device)
                labels = self.labels.clone().to(device)

                # Normalize prototypes
                prototypes = torch.nn.functional.normalize(prototypes, dim=1)
                #print('prototypes',prototypes.shape)

                # create similarity matrix between batch embeddings & prototypes
                z_center_similarity_matrix_1 = torch.einsum(
                    "nd,md->nm", z1, prototypes)
                z_center_similarity_matrix_2 = torch.einsum(
                    "nd,md->nm", z2, prototypes)
                #print('z_center_similarity_matrix_1',z_center_similarity_matrix_1.shape)

                # find nearest prototypes for each batch embedding
                _, topk_clusters_1 = torch.topk(
                    z_center_similarity_matrix_1, 1, dim=1)
                _, topk_clusters_2 = torch.topk(
                    z_center_similarity_matrix_2, 1, dim=1)

                z1_final = z1.clone()
                z2_final = z2.clone()
                # for each batch embedding
                for i in range(topk_clusters_1.shape[0]):

                    clusters_1 = topk_clusters_1[i, :]
                    clusters_2 = topk_clusters_2[i, :]

                    # find memory embedding indices that belong to the selected
                    # nearest cluster/prototype (for each view)
                    indices_1 = (labels[..., None] ==
                                 clusters_1).any(-1).nonzero().squeeze()
                    indices_2 = (labels[..., None] ==
                                 clusters_2).any(-1).nonzero().squeeze()
                    #print('indices_1',indices_1.nelement())
                   # print('indices_2',indices_2.nelement())

                    if indices_1.nelement() == 0:
                        # sanity check that all of the selected clusters more
                        # than 0 assigned memory embeddings
                        _, topk_indices_1 = torch.topk(
                            similarity_matrix1[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding &
                        # selected partition embeddings
                        tmp = bank_normed[indices_1, :].unsqueeze(
                            0) if indices_1.nelement() == 1 else bank_normed[indices_1, :]
                        z_memory_similarity_matrix_1 = torch.einsum(
                            "nd,md->nm", z1[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN partition embeddings (for each view)
                        if z_memory_similarity_matrix_1.dim() < 2:
                            _, topk_indices_1 = torch.topk(
                                similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        elif z_memory_similarity_matrix_1.shape[1] <= k:
                            _, topk_indices_1 = torch.topk(
                                similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        else:
                            _, topk_indices_1 = torch.topk(
                                z_memory_similarity_matrix_1, k, dim=1)

                    if indices_2.nelement() == 0:
                        # sanity check that all of the selected clusters more
                        # than 0 assigned memory embeddings
                        _, topk_indices_2 = torch.topk(
                            similarity_matrix2[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding &
                        # selected partition embeddings
                        tmp = bank_normed[indices_2, :].unsqueeze(
                            0) if indices_2.nelement() == 1 else bank_normed[indices_2, :]
                        z_memory_similarity_matrix_2 = torch.einsum(
                            "nd,md->nm", z2[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN partition embeddings (for each view)
                        if z_memory_similarity_matrix_2.dim() < 2:
                            _, topk_indices_2 = torch.topk(
                                similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        elif z_memory_similarity_matrix_2.shape[1] < k:
                            _, topk_indices_2 = torch.topk(
                                similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        else:
                            _, topk_indices_2 = torch.topk(
                                z_memory_similarity_matrix_2, k, dim=1)

                    if topk_indices_1.dim() < 2:
                        topk_indices_1 = topk_indices_1.unsqueeze(0)
                    if topk_indices_2.dim() < 2:
                        topk_indices_2 = topk_indices_2.unsqueeze(0)

                    # concat topk NN embeddings to original embeddings for each view
                    
                    for j in range(k):
                        z1_final = torch.cat((z1_final, torch.index_select(
                            bank.T, dim=0, index=topk_indices_1[:, j])), 0)
                        z2_final = torch.cat((z2_final, torch.index_select(
                            bank.T, dim=0, index=topk_indices_2[:, j])), 0)
                        indexs1 = torch.cat((indexs1, torch.index_select(
                            index_bank, dim=0, index=topk_indices_1[:, j])), 0)
                        indexs2 = torch.cat((indexs2, torch.index_select(
                            index_bank, dim=0, index=topk_indices_2[:, j])), 0)                        
                        
                        

                # concat the embeddings of the 2 views
                z = torch.cat((z1_final, z2_final), 0)
                bindexs = torch.cat((indexs1, indexs2), 0)
                #print('z',z.shape)
            else:
                # will only occur if the protoypes are initialized before the
                # memory is full for the first time (in practice not possible)

                # find indices of topk NN memory embeddings for each view
                _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
                _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

                # concat topk NN embeddings to original embeddings for each view
                for i in range(k):
                    z1 = torch.cat((z1, torch.index_select(
                        bank.T, dim=0, index=topk_indices_1[:, i])), 0)
                    z2 = torch.cat((z2, torch.index_select(
                        bank.T, dim=0, index=topk_indices_2[:, i])), 0)
                    indexs1 = torch.cat((indexs1, torch.index_select(
                            index_bank, dim=0, index=topk_indices_1[:, j])), 0)
                    indexs2 = torch.cat((indexs2, torch.index_select(
                            index_bank, dim=0, index=topk_indices_2[:, j])), 0) 
                # concat the embeddings of the 2 views
                z = torch.cat((z1, z2), 0)
                bindexs = torch.cat((indexs1, indexs2), 0)
            return z,bindexs
        else:
            return output,bindexs


def clusterer(z, algo='kmeans', n_clusters=5, metric='euclidean', hdb_min_cluster_size=4):
    
    predicted_labels = None
    probs = None
    if algo == 'kmeans':
        clf = cluster.KMeans(n_clusters=n_clusters, n_init=10)
        predicted_labels = clf.fit_predict(z)
   
    return clf, predicted_labels, probs
