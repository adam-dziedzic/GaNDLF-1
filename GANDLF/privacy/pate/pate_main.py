import numpy as np

from .rdp_cumulative import analyze_multiclass_confident_gnmax


def query_multiclass(votes: np.ndarray, threshold: float,
                     sigma_threshold: float, sigma_gnmax: float, delta: float,
                     budget: float = None):
    max_num_queries, dp_eps, _, _, _ = analyze_multiclass_confident_gnmax(
        votes=votes, threshold=threshold, sigma_threshold=sigma_threshold,
        sigma_gnmax=sigma_gnmax, delta=delta, budget=budget)
    dp_eps = dp_eps[-1]

    """Query a noisy ensemble model."""
    data_size = votes.shape[0]
    num_classes = votes.shape[1]
    # Thresholding mechanism (GNMax)
    if sigma_threshold > 0:
        noise_threshold = np.random.normal(
            loc=0.0, scale=sigma_threshold, size=data_size)
        vote_counts = votes.max(axis=-1)
        answered = (vote_counts + noise_threshold) > threshold
    else:
        answered = [True for _ in range(data_size)]
    # Gaussian mechanism
    assert sigma_gnmax > 0
    noise_gnmax = np.random.normal(0., sigma_gnmax, (
        data_size, num_classes))
    noisy_votes = votes + noise_gnmax
    preds = noisy_votes.argmax(axis=1).astype(np.int64)
    # Gap between the ensemble votes of the two most probable classes.
    # Sort the votes in descending order.
    sorted_votes = np.flip(np.sort(votes, axis=1), axis=1)
    # Compute the gap between 2 votes with the largest counts.
    gaps = (sorted_votes[:, 0] - sorted_votes[:, 1])

    return preds, gaps, answered, dp_eps, max_num_queries
