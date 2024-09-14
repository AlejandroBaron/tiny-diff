from torch import Tensor
from torch import distributions as dist


def mv_normal_log_likelihood(x: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
    """Multivariate normal log likelihood."""
    mvn = dist.MultivariateNormal(mu, covariance_matrix=cov)
    log_likelihood = mvn.log_prob(x)
    return -log_likelihood
