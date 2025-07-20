
def wei_log_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max, max_i = torch.max(in_features, dim=dim, keepdim = True)
    exp = torch.exp(in_features - max)
    log_exp = in_features - max
    sum = torch.sum(exp, dim=dim, keepdim = True)
    # log_softmax = log(exp / sum) = log(exp) - log(sum)
    return log_exp - torch.log(sum)


def wei_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    log_softmax = wei_log_softmax(inputs, dim=-1)
    pro = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(0)
    return torch.mean(-pro)
