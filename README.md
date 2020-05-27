# Description of files

This repository stores the files corresponding to our paper [Learning to Recognize Discontiguous
Entities](https://www.aclweb.org/anthology/D16-1008/).

`count_graph.py`: Counts the number of (hyper-)paths from the root to the leaves for a sequence of length `n`.

`-k` is the number of components in the discontiguous entity. So "pupils ... dilated" would have k=2

`-n` is the sequence length, which is the number of tokens in the sentence

`-v` is the verbose mode, allows you to see the transition matrix

For example, to produce the transition matrix in the supplementary material, use:

    python count_graph.py -v -k 2 --shared

(This will also print the number of paths on a sequence of length n=4 by default)

The corresponding paper can be read at: https://www.aclweb.org/anthology/D16-1008/ or at arXiv:
https://arxiv.org/abs/1810.08579

Note that the reference to Tang et al. in the ACL Anthology points to an incorrect version. The arXiv version fixes
this mistake.

To cite the code, please use the following:

    Aldrian Obaja Muis and Wei Lu. 2016. Learning to Recognize Discontiguous Entities. Proceedings of the 2016
    Conference on Empirical Methods in Natural Language Processing, 75–84. https://doi.org/10.18653/v1/D16-1008.
    Supplementary Material. Code retrieved from 
