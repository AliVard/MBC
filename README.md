# MBC
Mixture-Based Correction (CIKM'21)

Please cite this if you use this code:

Ali Vardasbi, Maarten de Rijke, and Ilya Markov. 2021. Mixture-Based Correction for Position and Trust Bias in Counterfactual Learning to Rank.
In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM ’21), November 1–5, 2021, Virtual Event, QLD, Australia. ACM, New York, NY, USA. https://doi.org/10.1145/3459637.3482275


We propose a new correction method for position and trust bias in \ac{CLTR} in which, unlike the existing methods, the correction does not rely on relevance estimation.
Our proposed method, MBC, is based on the assumption that the distribution of the \aclp{CTR} over the items being ranked is a mixture of two distributions: the distribution of \aclp{CTR} for relevant items and the distribution of \aclp{CTR} for non-relevant items.
We prove that our method is unbiased.
The validity of our proof is not conditioned on accurate bias parameter estimation.
Our experiments show that MBC, when used in different bias settings and accompanied by different \acl{LTR} algorithms, outperforms \ac{AC}, the state-of-the-art method for correcting position and trust bias, in some settings, while performing on par in other settings.
Furthermore, MBC is orders of magnitude more efficient than \ac{AC} in terms of the training time.
