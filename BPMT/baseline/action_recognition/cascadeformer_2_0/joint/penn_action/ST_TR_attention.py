import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F_func
import math


class Unit2D(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dim=2,
                 dropout=0,
                 bias=True):
        super(Unit2D, self).__init__()
        pad = int((kernel_size - 1) / 2)
        print("Pad Temporal ", pad)

        if dim == 2:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
                bias=bias)
        elif dim == 3:
            print("Pad Temporal ", pad)
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                stride=(1, stride),
                bias=bias)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)

        # initialize
        conv_init(self.conv)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.bn(self.conv(x)))
        return x


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n = n*k
    module.weight.data.normal_(0, math.sqrt(2. / n))


'''
    Adapted from ST-TR (https://github.com/Chiaraplizz/ST-TR/blob/master/code/st_gcn/net/spatial_transformer.py)
'''

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
multi_matmul = False
dropout = False
scale_norm = False
save = False


class spatial_attention(nn.Module):
    def __init__(self, in_channels, kernel_size, dk, dv, Nh, complete, relative, layer, A, more_channels, drop_connect,
                 adjacency, num, num_point,
                 shape=25, stride=1,
                 last_graph=False, data_normalization=True, skip_conn=True, visualization=True):
        super(spatial_attention, self).__init__()
        self.in_channels = in_channels
        self.complete = complete
        self.kernel_size = 1
        self.dk = dk
        self.dv = dv
        self.num = num
        self.layer = layer
        self.more_channels = more_channels
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.adjacency = adjacency
        self.Nh = Nh
        self.num_point=num_point
        self.A = A[0] + A[1] + A[2]
        if self.adjacency:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        self.shape = shape
        self.relative = relative
        self.last_graph = last_graph
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."


        if (self.more_channels):

            self.qkv_conv = nn.Conv2d(self.in_channels, (2 * self.dk + self.dv) * self.Nh // self.num,
                                      kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding)
        else:
            self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding)
        if (self.more_channels):

            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            # Two parameters are initialized in order to implement relative positional encoding
            # One weight repeated over the diagonal
            # V^2-V+1 paramters in positions outside the diagonal
            if self.more_channels:
                self.key_rel = nn.Parameter(torch.randn(((self.num_point ** 2) - self.num_point, self.dk // self.num), requires_grad=True))
            else:
                self.key_rel = nn.Parameter(torch.randn(((self.num_point ** 2) - self.num_point, self.dk // Nh), requires_grad=True))
            if self.more_channels:
                self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.num), requires_grad=True))
            else:
                self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.Nh), requires_grad=True))

    def forward(self, x, label, name):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems
        if (multi_matmul):
            for i in range(0, 5):
                flat_q_5 = flat_q[:, :, :, (5 * i):(5 * (i + 1))]
                product = torch.matmul(flat_q_5.transpose(2, 3), flat_k)
                if (i == 0):
                    logits = product
                else:
                    logits = torch.cat((logits, product), dim=2)
        else:
            logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        # In this version, the adjacency matrix is weighted and added to the attention logits of transformer to add
        # information of the original skeleton structure
        if (self.adjacency):
            self.A = self.A.cuda(device)
            logits = logits.reshape(-1, V, V)
            M, V, V = logits.shape
            A = self.A
            A *= self.mask
            A = A.unsqueeze(0).expand(M, V, V)
            logits = logits+A
            logits = logits.reshape(B, self.Nh, V, V)

        # Relative positional encoding is used or not
        if self.relative:
            rel_logits = self.relative_logits(q)
            logits_sum = torch.add(logits, rel_logits)

        # Calculate weights
        if self.relative:
            weights = F.softmax(logits_sum, dim=-1)
        else:
            weights = F.softmax(logits, dim=-1)

        # Drop connect implementation to avoid overfitting
        if (self.drop_connect and self.training):
            mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * V, device=device))
            mask = mask.reshape(B, self.Nh, V).unsqueeze(2).expand(B, self.Nh, V, V)
            weights = weights * mask
            weights = weights / (weights.sum(3, keepdim=True) + 1e-8)



        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        if not self.more_channels:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
        else:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.num))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        # if self.more_channels=True, to each head is assigned dk*self.Nh//self.num channels
        if self.more_channels:
            q, k, v = torch.split(qkv, [dk * self.Nh // self.num, dk * self.Nh // self.num, dv * self.Nh // self.num],
                                  dim=1)
        else:
            q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q*(dkh ** -0.5)
        if self.more_channels:
            flat_q = torch.reshape(q, (N, Nh, dk // self.num, T * V))
            flat_k = torch.reshape(k, (N, Nh, dk // self.num, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.num, T * V))
        else:
            flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
            flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, T, V = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
        q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))

        # q used to multiply for the embedding of the parameter on the diagonal
        q = torch.reshape(q, (B * Nh * T, V, dk))
        # key_rel_diagonal: (1, dk) -> (V, dk)
        param_diagonal = self.key_rel_diagonal.expand((V, dk))
        rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
        return rel_logits

    def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
        # compute relative logits along one dimension
        # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)

        # (B*Nh*1, V^2-V)
        rel_logits = torch.einsum('bmd,md->bm', q_first, rel_k)
        # (B*Nh*1, V)
        rel_logits_diagonal = torch.einsum('bmd,md->bm', q, param_diagonal)

        # reshapes to obtain Srel
        rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
        return rel_logits

    def rel_to_abs(self, rel_logits, rel_logits_diagonal):
        B, L = rel_logits.size()
        B, V = rel_logits_diagonal.size()

        # (B, V-1, V) -> (B, V, V)
        rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
        row_pad = torch.zeros(B, 1, V).to(rel_logits)
        rel_logits = torch.cat((rel_logits, row_pad), dim=1)

        # concat the other embedding on the left
        # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
        rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
        rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
        rel_logits = torch.reshape(rel_logits, (B, V + 1, V))

        # slice
        flat_sliced = rel_logits[:, :V, :]
        final_x = torch.reshape(flat_sliced, (B, V, V))
        return final_x
    

''' 
    Adapted from ST-TR (https://github.com/Chiaraplizz/ST-TR/blob/master/code/st_gcn/net/temporal_transformer.py)
'''


class tcn_unit_attention(nn.Module):
    def __init__(self, in_channels, out_channels, dv_factor, dk_factor, Nh, n, frames,
                 relative, only_temporal_attention, dropout, kernel_size_temporal, stride, weight_matrix,
                 last, layer, device, more_channels, drop_connect, num_point,
                 bn_flag=True,
                 shape=25, visualization=False, data_normalization=True, skip_conn=True, more_relative=False):
        super(tcn_unit_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = layer
        self.visualization = visualization
        self.num_point = num_point
        self.more_channels = more_channels
        self.only_temporal_att = only_temporal_attention
        self.drop_connect = drop_connect
        self.kernel_size_temporal = kernel_size_temporal
        self.num = n
        self.more_relative = more_relative
        self.dk = int(dk_factor * out_channels)
        self.Nh = Nh
        self.bn_flag = bn_flag
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size_temporal - 1) // 2
        self.bn = nn.BatchNorm2d(out_channels)
        self.weight_matrix = weight_matrix
        self.relu = nn.ReLU(inplace=True)
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.last = last

        if (not self.only_temporal_att):
            self.dv = int(dv_factor * out_channels)
        else:
            self.dv = out_channels
        if ((self.in_channels != self.out_channels) or (stride != 1)):
            self.down = Unit2D(
                self.in_channels, self.out_channels, kernel_size=1, stride=stride)
        else:
            self.down = None
        if self.data_normalization:
            #self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
            self.data_bn = nn.BatchNorm1d(in_channels)
        if dropout:
            self.dropout = nn.Dropout(0.25)

        # Temporal convolution
        if (not self.only_temporal_att):
            self.tcn_conv = Unit2D(in_channels, out_channels - self.dv, dropout=dropout,
                                   kernel_size=kernel_size_temporal,
                                   stride=self.stride)
        if (self.more_channels):

            self.qkv_conv = nn.Conv2d(self.in_channels, (2 * self.dk + self.dv) * self.Nh // self.num,
                                      kernel_size=(1, stride),
                                      stride=(1, stride),
                                      padding=(0, int((1 - 1) / 2)))
        else:
            if self.num_point % 2 != 0:
                self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=(1, stride),
                                      stride=(1, stride),
                                      padding=(0, int((1 - 1) / 2)))
            else:
                self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=(1, 1),
                                          stride=(1, stride),
                                          padding=(0, int((1 - 1) / 2)))
        if (self.more_channels):
            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        # if self.out_channels == 64:
        #     frames = 300

        # if self.out_channels == 128:
        #     frames = 150

        # if self.out_channels == 256:
        #     frames = 75
        # else:
        #     frames = input_seq_len

        if self.relative:
            if self.more_channels:
                self.key_rel = nn.Parameter(
                    torch.randn((2 * frames - 1, self.dk // self.num), requires_grad=True))

            else:
                self.key_rel = nn.Parameter(
                    torch.randn((2 * frames - 1, self.dk // Nh), requires_grad=True))

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"

    def forward(self, x):
        # Input x
        # (batch_size, channels, time, joints)
        N, C, T, V = x.size()
        x_sum = x

        if (self.data_normalization):
            x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)

            x = x.squeeze(-1)      # (B, D, T)
            x = self.data_bn(x)    # ✅ Now valid for BatchNorm1d
            x = x.unsqueeze(-1)    # (B, D, T, 1) — restore shape

            #x = self.data_bn(x)

            x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        # Joint dimension is put inside the batch, in order to process each joint along the time separately
        x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T)

        if scale_norm:
            self.scale = ScaleNorm(scale=C ** 0.5)
            x = self.scale(x)

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        B, self.Nh, C, T = flat_q.size()

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, time, dkh)*(batch_size, Nh,dkh, time) =  (batch_size, Nh, time, time)

        if (multi_matmul):
            for i in range(0, 5):
                flat_q_5 = flat_q[:, :, :, (60 * i):(60 * (i + 1))]
                product = torch.matmul(flat_q_5.transpose(2, 3), flat_k)
                if (i == 0):
                    logits = product
                else:
                    logits = torch.cat((logits, product), dim=2)
        else:
            logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            rel_logits = self.relative_logits(q)
            logits_sum = torch.add(logits, rel_logits)

        # Calculate weights
        if self.relative:

            weights = F_func.softmax(logits_sum, dim=-1)

        else:
            weights = F_func.softmax(logits, dim=-1)

        if (self.drop_connect and self.training):
            mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * T, device=device))
            mask = mask.reshape(B, self.Nh, T).unsqueeze(2).expand(B, self.Nh, T, T)
            weights = weights * mask
            weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

        # attn_out
        # (batch, Nh, time, dvh)
        # weights*V
        # (batch, Nh, time, time)*(batch, Nh, time, dvh)=(batch, Nh, time, dvh)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        if not self.more_channels:
            attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, self.dv // self.Nh))
        else:
            attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, self.dv // self.num))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, time, 1)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, time, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        attn_out = attn_out.reshape(N, V, -1, T).permute(0, 2, 3, 1)

        if self.skip_conn:
            if dropout:
                attn_out = self.dropout(attn_out)

                if (not self.only_temporal_att):
                    x = self.tcn_conv(x_sum)
                    result = torch.cat((x, attn_out), dim=1)
                else:
                    result = attn_out

                result = result+(x_sum if (self.down is None) else self.down(x_sum))


            else:
                if (not self.only_temporal_att):
                    x = self.tcn_conv(x_sum)
                    result = torch.cat((x, attn_out), dim=1)
                else:
                    result = attn_out

                result = result+(x_sum if (self.down is None) else self.down(x_sum))


        else:
            result = attn_out

        if (self.bn_flag):
            result = self.bn(result)
        result = self.relu(result)
        return result

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)

        # In this case V=1, because Temporal Transformer is applied for each joint separately
        N, C, V1, T1 = qkv.size()
        if self.more_channels:
            q, k, v = torch.split(qkv, [dk * self.Nh // self.num, dk * self.Nh // self.num, dv * self.Nh // self.num],
                                  dim=1)
        else:
            q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)

        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q* (dkh ** -0.5)
        if self.more_channels:

            flat_q = torch.reshape(q, (N, Nh, dk // self.num, V1 * T1))
            flat_k = torch.reshape(k, (N, Nh, dk // self.num, V1 * T1))
            flat_v = torch.reshape(v, (N, Nh, dv // self.num, V1 * T1))
        else:
            flat_q = torch.reshape(q, (N, Nh, dkh, V1 * T1))
            flat_k = torch.reshape(k, (N, Nh, dkh, V1 * T1))
            flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, V1 * T1))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, F, V = x.size()
        ret_shape = (B, Nh, channels // Nh, F, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, F, V = x.size()
        ret_shape = (batch, Nh * dv, F, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, _, T = q.size()
        # B, Nh, V, T, dk -> B, Nh, F, 1, dk
        q = q.permute(0, 1, 3, 4, 2)
        q = q.reshape(B, Nh, T, dk)
        rel_logits = self.relative_logits_1d(q, self.key_rel)
        return rel_logits

    def relative_logits_1d(self, q, rel_k):
        # compute relative logits along one dimension
        # (B, Nh,  1, V, channels // Nh)*(2 * K - 1, self.dk // Nh)
        # (B, Nh,  1, V, 2 * K - 1)
        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        B, Nh, L, L = rel_logits.size()
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        print(x.shape)
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = scale

        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        return x * norm