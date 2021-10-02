# Evalution-metrices-for-Binary-Segmentations

# 1-Aggregated Jaccard Index

AJI (Aggregated Jaccard Index) is a concept proposed in 2017. Source of the paper: 
A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology 
This is a paper on the MICCAI18 cell segmentation data set. 
AJI can be said to be an enhanced version of IOU, but compared to IOU, he has a stronger ability to measure the effect of strength division. Why **, because it is based on connected domains, not pixel-based **. The following is a detailed explanation of how this AJI is calculated. 
The first is to give the formula:

(https://user-images.githubusercontent.com/76598242/135724132-4867dd84-c5b3-4d74-aaf8-66269c2ce157.png)

please before getting dive into code have a look to this link(https://blog-csdn-net.translate.goog/qq_34616741/article/details/103399675?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=nui,scto) to understand the idea and the differance between it and IOU.:
