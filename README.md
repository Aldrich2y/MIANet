# MIANet
Official PyTorch Implementation of MIANet: Aggregating Unbiased Instance and General Information for Few-Shot Semantic Segmentation(CVPR 2023).


> **Abstract**: *Existing few-shot segmentation methods are based on the meta-learning strategy and extract instance knowledge
from a support set and then apply the knowledge to segment target objects in a query set. However, the extracted
knowledge is insufficient to cope with the variable intraclass differences since the knowledge is obtained from a
few samples in the support set. To address the problem,
we propose a multi-information aggregation network (MIANet) that effectively leverages the general knowledge, i.e.,
semantic word embeddings, and instance information for
accurate segmentation. Specifically, in MIANet, a general
information module (GIM) is proposed to extract a general
class prototype from word embeddings as a supplement to
instance information. To this end, we design a triplet loss
that treats the general class prototype as an anchor and
samples positive-negative pairs from local features in the
support set. The calculated triplet loss can transfer semantic similarities among language identities from a word embedding space to a visual representation space. To alleviate the model biasing towards the seen training classes
and to obtain multi-scale information, we then introduce a
non-parametric hierarchical prior module (HPM) to generate unbiased instance-level information via calculating the
pixel-level similarity between the support and query image
features. Finally, an information fusion module (IFM) combines the general and instance information to make predictions for the query image. Extensive experiments on
PASCAL-5i and COCO-20i
show that MIANet yields superior performance and set a new state-of-the-art.*


