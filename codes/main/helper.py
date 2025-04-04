from scipy import stats

def select_statistical_test(group1, group2, logger):
    """
    Select appropriate statistical test based on sample size and normality.
    Will use t-test when normality assumptions are met, otherwise Mann-Whitney U.

    Parameters:
    -----------
    group1 : array-like
        First group of observations
    group2 : array-like
        Second group of observations
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Dictionary containing test type, statistic, p-value, and any other relevant info
    """
    # Check minimum sample size for reliable analysis
    if len(group1) < 2 or len(group2) < 2:
        logger.warning("Insufficient sample size for statistical testing (n<2 in at least one group)")
        return {
            'test_type': "not performed",
            'test_statistic': None,
            'p_value': None,
            'reason': "insufficient sample size"
        }

    # Check if samples are large enough for normality testing
    group1_large_enough = len(group1) >= 8
    group2_large_enough = len(group2) >= 8

    # Test for normality if samples are large enough
    if group1_large_enough:
        _, p_norm1 = stats.shapiro(group1)
        group1_normal = p_norm1 > 0.05
        logger.info(f"Group 1: {'normally' if group1_normal else 'not normally'} distributed (p={p_norm1:.4f})")
    else:
        group1_normal = None
        logger.warning(f"Group 1: Too few data points for reliable normality test")

    if group2_large_enough:
        _, p_norm2 = stats.shapiro(group2)
        group2_normal = p_norm2 > 0.05
        logger.info(f"Group 2: {'normally' if group2_normal else 'not normally'} distributed (p={p_norm2:.4f})")
    else:
        group2_normal = None
        logger.warning(f"Group 2: Too few data points for reliable normality test")

    # Decide on test based on normality results
    # Use t-test only when both groups are confirmed to be normally distributed
    can_use_ttest = (group1_normal is True and group2_normal is True)

    # For small samples where normality can't be reliably tested,
    # default to non-parametric test
    if group1_normal is None or group2_normal is None:
        can_use_ttest = False

    # Perform appropriate test
    if can_use_ttest:
        # Use t-test for normally distributed data
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        logger.info(f"Selected test: t-test (t={t_stat:.2f}, p={p_value:.4f})")
        return {
            'test_type': "t-test",
            'test_statistic': t_stat,
            'p_value': p_value,
            'reason': "both groups normally distributed"
        }
    else:
        # Use Mann-Whitney U test for non-normal data
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        logger.info(f"Selected test: Mann-Whitney U test (U={u_stat:.2f}, p={p_value:.4f})")
        return {
            'test_type': "Mann-Whitney-U",
            'test_statistic': u_stat,
            'p_value': p_value,
            'reason': "at least one group not normally distributed or small sample size"
        }