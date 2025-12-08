import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGATLayer(nn.Module):
    def __init__(self, n_heads, in_features, out_features, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            GATELayer(in_features, out_features, alpha) for _ in range(n_heads)
        ])
        self.concat = concat  # True for intermediate layers, False for last layer

    def forward(self, x, adj, M):
        if self.concat:
            # 拼接所有头的输出 (dim=2 最后是 [N, n_heads * out_features])
            out = torch.cat([head(x, adj, M, concat=True) for head in self.heads], dim=1)
        else:
            # 平均所有头的输出 (适用于最后一层)
            out = torch.mean(torch.stack([head(x, adj, M, concat=False) for head in self.heads]), dim=0)
        return out

class GATELayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        jihuo = 1.414
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=jihuo)
        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=jihuo)
        nn.init.xavier_uniform_(self.a_neighs.data, gain=jihuo)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _align_M_values_to_adj(self, adj_idx, M, N, device):
        """
        返回与 adj_idx 对齐的 M_values（长度 = adj 边数 E_a），全部在 device 上计算。
        adj_idx: LongTensor [2, E_a]
        M: sparse coalesced tensor or dense tensor
        """
        row_a, col_a = adj_idx[0], adj_idx[1]
        E_a = row_a.size(0)

        if not M.is_sparse:
            # 如果 M 是稠密矩阵（N x N），直接按边索引取值（支持 GPU）
            return M[row_a, col_a]  # [E_a]

        # M 是稀疏：做排序对齐（全部在 GPU）
        Mc = M.coalesce()
        m_idx = Mc.indices()   # [2, E_m]
        m_vals = Mc.values()   # [E_m]

        row_m, col_m = m_idx[0], m_idx[1]
        E_m = row_m.size(0)

        # 线性索引（N 可能较大，确保 long）
        lin_a = (row_a.to(torch.long) * N + col_a.to(torch.long)).to(device)  # [E_a]
        lin_m = (row_m.to(torch.long) * N + col_m.to(torch.long)).to(device)  # [E_m]

        # 排序两组线性索引
        sorted_a_idx = torch.argsort(lin_a)
        sorted_m_idx = torch.argsort(lin_m)

        lin_a_sorted = lin_a[sorted_a_idx]   # [E_a]
        lin_m_sorted = lin_m[sorted_m_idx]   # [E_m]

        # 对 M 的 values 做相同的重排
        m_vals_sorted = m_vals[sorted_m_idx]

        # 在 lin_m_sorted 中查找 lin_a_sorted 的位置（GPU 上的 searchsorted）
        pos = torch.searchsorted(lin_m_sorted, lin_a_sorted)

        # pos 可能等于 E_m（越界），clamp 一下便于索引再检测是否匹配
        pos_clamped = pos.clamp(max=E_m - 1)

        # 获取 candidate lin 值并比较是否真正匹配
        candidate = lin_m_sorted[pos_clamped]
        matched = candidate == lin_a_sorted  # boolean mask [E_a]

        # 从 m_vals_sorted 中取出对应值（pos_clamped），不匹配的设为 0
        candidate_vals = m_vals_sorted[pos_clamped]
        # 将不匹配的位置置为 0
        candidate_vals = candidate_vals * matched.to(candidate_vals.dtype)

        # 现在把值恢复到原始 adj 顺序
        M_values = torch.zeros(E_a, dtype=candidate_vals.dtype, device=device)
        M_values[sorted_a_idx] = candidate_vals

        return M_values  # [E_a]

    def forward(self, input, adj, M, concat=True):
        """
        input: [N, in_features]
        adj:   sparse tensor (coalesced recommended)
        M:     sparse tensor or dense tensor (coalesced if sparse)
        """
        device = input.device
        N = input.size(0)

        # 确保 adj coalesced（去重并排序），否则 indices/values 可能不对齐
        if adj.is_sparse:
            adj = adj.coalesce()

        if M.is_sparse:
            M = M.coalesce()

        edge_index = adj.indices()   # [2, E_a]
        # edge_weight = adj.values()  # 如需可用

        src, dst = edge_index[0], edge_index[1]  # both [E_a]

        # 节点线性变换
        h = torch.mm(input, self.W)  # [N, out_features]

        # 注意力基础项
        attn_self = torch.mm(h, self.a_self)    # [N,1]
        attn_neigh = torch.mm(h, self.a_neighs) # [N,1]

        # 计算每条边的 e_ij（按 adj 的边顺序）
        e_ij = attn_self[src] + attn_neigh[dst]   # [E_a, 1]

        # 取与 adj 对应的 M 值（可能是 sparse 或 dense）
        M_values = self._align_M_values_to_adj(edge_index, M, N, device)  # [E_a]

        # 应用 LeakyReLU 并乘以 M 的边权（若 M_values 包含 0 表示缺失或屏蔽）
        e_ij = self.leakyrelu(e_ij).squeeze(1) * M_values  # -> [E_a]

        # softmax 归一化（按目标节点 dst 分组）
        exp_e = torch.exp(e_ij)  # [E_a]
        # denom: 每个目标节点的归一化分母，形状 [N,1]，用 scatter_add 累加到 dst 上
        denom = torch.zeros(N, device=device, dtype=exp_e.dtype).scatter_add_(0, dst, exp_e)
        # 防止除零
        denom_safe = denom[dst] + 1e-16
        alpha = (exp_e / denom_safe).unsqueeze(1)  # [E_a,1]

        # 聚合：h_prime[j] += alpha_ij * h[i]
        h_prime = torch.zeros_like(h)
        # alpha * h[src] -> [E_a, out_feat], 然后按 dst 累加到节点上
        h_prime = h_prime.index_add_(0, dst, alpha * h[src])

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

# class GATELayer(nn.Module):
#     """
#     Simple GATE layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, alpha=0.2):
#         super(GATELayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         jihuo = 1.414
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         # Xavier均匀分布
#         # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
#         # 这里有一个gain，增益的大小是依据激活函数类型来设定
#         # eg：nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
#         #
#         # PS：上述初始化方法，也称为Glorot initialization
#         nn.init.xavier_uniform_(self.W.data, gain=jihuo)
#
#         self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         # print(self.a_self.data)
#         nn.init.xavier_uniform_(self.a_self.data, gain=jihuo)
#         # print(self.a_self.data)
#
#         self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         nn.init.xavier_uniform_(self.a_neighs.data, gain=jihuo)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         # print(self.leakyrelu)
#
#     def forward(self, input, adj, M, concat=True):
#         # torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
#         h = torch.mm(input, self.W)
#         # x: [samples_cnt=N, input_feat]
#         # w: [input_feat, output_feat]
#         # h: [N, output_feat]
#         attn_for_self = torch.mm(h, self.a_self)  # (N,1)
#         attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
#         # >>> a
#         # tensor([[1],
#         #         [2],
#         #         [3]])
#         # >>> torch.transpose(a, 0, 1)
#         # tensor([[1, 2, 3]])
#         # >>> a+torch.transpose(a, 0, 1)
#         # tensor([[2, 3, 4],
#         #         [3, 4, 5],
#         #         [4, 5, 6]])
#         attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)  # [N, N]
#         # [N, N]*[N, N]=>[N, N]
#         attn_dense = torch.mul(attn_dense, M)
#         # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
#         attn_dense = self.leakyrelu(attn_dense)  # (N,N)
#
#         zero_vec = -9e15 * torch.ones_like(adj)  # [N, N] #生成与邻接矩阵adj尺寸相似的矩阵
#         # torch.where: Return a tensor of elements selected from either x or y, depending on condition
#         # torch.where(condition, x, y) → Tensor, xi if condition else yi 如果adj>0 那么adj=attn_dense
#         adj = torch.where(adj > 0, attn_dense, zero_vec)  # [N, N]
#         # 对每一行的样本所有邻居softmax
#         attention = F.softmax(adj, dim=1)  # [N, N]
#         # attention: [N, N]
#         # h: [N, output_feat]
#         h_prime = torch.matmul(attention, h)  # N, output_feat
#
#         if concat:
#             # torch.nn.function.elu: Applies element-wise, ELU(x)=max(0,x)+min(0,α∗(exp(x)−1)) .
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + " ("
#             + str(self.in_features)
#             + " -> "
#             + str(self.out_features)
#             + ")"
#         )



class MultiHeadGATE(nn.Module):
    """
    Multi-head wrapper for GATELayer
    """

    def __init__(self, in_features, out_features, alpha=0.2, heads=4, concat=True):
        super(MultiHeadGATE, self).__init__()
        self.heads = heads
        self.concat = concat

        self.attentions = nn.ModuleList(
            [GATELayer(in_features, out_features, alpha) for _ in range(heads)]
        )

    def forward(self, x, adj, M):
        head_outs = [att(x, adj, M, concat=True) for att in self.attentions]

        if self.concat:
            # 拼接方式: 输出维度 = heads * out_features
            return torch.cat(head_outs, dim=1)
        else:
            # 平均方式: 输出维度 = out_features
            return torch.mean(torch.stack(head_outs), dim=0)