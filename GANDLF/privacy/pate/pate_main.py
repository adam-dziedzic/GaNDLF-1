import numpy as np

from privacy.pate.rdp_cumulative import analyze_multiclass_confident_gnmax


def query_multiclass(
    votes: np.ndarray,
    threshold: float,
    sigma_threshold: float,
    sigma_gnmax: float,
    delta: float,
    budget: float = None,
) -> (np.ndarray, np.ndarray, np.ndarray, int, np.ndarray):
    """
    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
            corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism from
            PATE confident GNMax.
        sigma_threshold: std (standard deviation) of the Gaussian noise in the
        threshold mechanism from the PATE confident GNMax.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        delta: pre-defined delta value for (eps, delta)-DP.
        budget: pre-defined epsilon value for (eps, delta)-DP, if assign None,
            then we try to answer all queries and report the final epsilon
            budget.

    Returns:
        predictions: noisy predictions from PATE. For each sample/query in the
            input votes, we give the answer predicted by PATE via noisy argmax.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the boolean value if a given query was answered or
            not.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the cumulative privacy cost epsilon.
        max_num_query: when the pre-defined privacy budget is set and exhausted.
        gaps: difference between number of votes for the winning class vs the
            runner-up (the 2nd class with the largest number of votes.
    """
    max_num_queries, dp_eps, _, _, _ = analyze_multiclass_confident_gnmax(
        votes=votes,
        threshold=threshold,
        sigma_threshold=sigma_threshold,
        sigma_gnmax=sigma_gnmax,
        delta=delta,
        budget=budget,
    )

    """Query a noisy ensemble model."""
    data_size = votes.shape[0]
    num_classes = votes.shape[1]
    # Thresholding mechanism (GNMax)
    if threshold > 0:
        assert sigma_threshold > 0
        noise_threshold = np.random.normal(
            loc=0.0, scale=sigma_threshold, size=data_size
        )
        vote_counts = votes.max(axis=-1)
        answered = (vote_counts + noise_threshold) > threshold
    else:
        assert threshold <= 0
        assert sigma_threshold <= 0
        answered = [True for _ in range(data_size)]
    # Gaussian mechanism
    assert sigma_gnmax > 0
    noise_gnmax = np.random.normal(0.0, sigma_gnmax, (data_size, num_classes))
    noisy_votes = votes + noise_gnmax
    predictions = noisy_votes.argmax(axis=1).astype(np.int64)
    # Gap between the ensemble votes of the two most probable classes.
    # Sort the votes in descending order.
    sorted_votes = np.flip(np.sort(votes, axis=1), axis=1)
    # Compute the gap between 2 votes with the largest counts.
    gaps = sorted_votes[:, 0] - sorted_votes[:, 1]

    return predictions, answered, dp_eps, max_num_queries, gaps


if __name__ == "__main__":
    votes = np.array(
        [
            [250] + [0] * 9,  # votes only for a single class
            [125, 125] + [0] * 8,  # 2 classes with equal number of votes
            [50] * 5 + [0] * 5,  # 5 classes with equal number of votes
            [25] * 10,  # votes evenly distributed between classes
        ]
    )

    print("PATE with the confident GNMax.")
    predictions, answered, dp_eps, max_num_queries, gaps = query_multiclass(
        votes=votes, threshold=200, sigma_threshold=150, sigma_gnmax=40, delta=10e-5
    )

    print("predictions: ", predictions)
    print("answered: ", answered)
    print("dp_eps cumulative: ", dp_eps)
    print("final epsilon: ", dp_eps[-1])
    print("max_num_queries: ", max_num_queries)
    print("gaps: ", gaps)

    print("PATE without the confident GNMax.")
    predictions, answered, dp_eps, max_num_queries, gaps = query_multiclass(
        votes=votes, threshold=0, sigma_threshold=0, sigma_gnmax=20, delta=10e-5
    )

    print("predictions: ", predictions)
    print("answered: ", answered)
    print("dp_eps cumulative: ", dp_eps)
    print("final epsilon: ", dp_eps[-1])
    print("max_num_queries: ", max_num_queries)
    print("gaps: ", gaps)
