# MIANet
**Official PyTorch Implementation of MIANet: Aggregating Unbiased Instance and General Information for Few-Shot Semantic Segmentation(CVPR 2023) [](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_MIANet_Aggregating_Unbiased_Instance_and_General_Information_for_Few-Shot_Semantic_CVPR_2023_paper.pdf)**
<img src="/figure/logo.png"  width="1000" height="250" align="bottom" />


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
PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>
show that MIANet yields superior performance and set a new state-of-the-art.*

![pipeiline](/figure/pipeline.png "The pipleline of MIANet")

## &#x1F527;Get Started
**Just follow these steps to train and test MIANet.**
### Dataset (Following the BAM operations)
**1.** Download the dataset from the following links.
+ PASCAL-5<sup>i</sup>: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
+ COCO-20<sup>i</sup>: [MSCOCO2014](https://cocodataset.org/#download)
  
**2.** Adjust these files to the following directory:    

                                                 JPEGImages
                     VOCdevkit2012--> VOC2012-->
                                                 SegmentationClassAug
      MIANet/data---->           
                                    annotations
                     MSCOCO2014-->  train2014  
                                    val2014

**3.** Download the data [lists](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/ESvJvL7X86pNqK5LSaKwK0sByDLwNx0kh73PVJJ_m1vSCg?e=RBjfKp) (.txt files) and put them into the MIANet/lists_bam directory.

### Models 
(Pretrained PSPNet) Download the pre-trained PSPNet from [here](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/ETERT3xe5ndEpDhStts7JmcBlYDY_2G0hPVJUBtLLG-njg?e=MLzVIL) and put them under initmodel/pascal and initmodel/coco directory. 

### Scripts
**1.** Change configuration via the .yaml files in MIANet/config.   
**2.** Train MIANet under the settings (resnet50 backbone; fold=0; shot=1):   run  *train.py*.   
**3.** Test MIANet (fill the *weight* parameter in .yaml file): run *test.py*

### Visualization
![result](/figure/result.png)

### To-Do List
- [ ] Complete the MIANet repository.

### References
This repo is mainly built based on PFENet, BAM. Thanks for their great work!

