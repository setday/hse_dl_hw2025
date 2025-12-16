#############################################
# Implementation of NeuralNDCG from allRank #
# but with some modifications, that is not  #
# connected with the logic of the loss.     #
#############################################


import torch


def dcg(y_pred, y_true, ats, gain_function=lambda x: torch.pow(2, x) - 1):
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    ats = [min(at, actual_length) for at in ats]

    _, indices = y_pred.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices)

    positions = torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float, device=true_sorted_by_preds.device)
    discounts = (1.0 / torch.log2(positions + 2.0))

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def sinkhorn_scaling(mat, tol=1e-6, max_iter=50):
    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-8)

        if torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    return mat


def deterministic_neural_sort(s, tau):
    dev = s.device

    b, n, *_ = s.size()
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    A_s = torch.abs(s - s.permute(0, 2, 1))

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - 1 - 2 * torch.arange(n, device=dev) for _ in range(b)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def stochastic_neural_sort(s, n_samples, tau, beta=1.0, log_scores=True, eps=1e-8):
    b, n, *_ = s.size()
    s_positive = s + torch.abs(s.min())
    U = torch.rand([n_samples, b, n, 1], device=s.device)
    samples = -beta * torch.log(-torch.log(U + eps) + eps)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * b, n, 1)

    P_hat = deterministic_neural_sort(s_perturb, tau)
    P_hat = P_hat.view(n_samples, b, n, n)
    return P_hat


def neuralNDCG(y_pred, y_true, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = y_pred.device

    if k is None:
        k = y_true.shape[1]

    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    y_true_masked = y_true.unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-8)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG
