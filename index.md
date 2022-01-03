##  Rotation Axis Focused Attention Network (RAFA-Net) for Estimating Head Pose
**Ardhendu Behera, Zachary Wharton, Pradeep Hewage and Swagat Kumar**<br>
**_Department of Computer Science, Edge Hill University, United Kingdom_**

### Abstract
Head pose is a vital indicator of human attention and behavior. Therefore, automatic estimation of head pose from images is key to many real-world applications. In this paper, we propose a novel approach for head pose estimation from a single RGB image. Many existing approaches often predict head poses by localizing facial landmarks and then solve 2D to 3D correspondence problem with a mean head model. Such approaches completely rely on the landmark detection accuracy, an ad-hoc alignment step, and the extraneous head model. To address this drawback, we present an end-to-end deep network, which explores rotation axis (yaw, pitch, and roll) focused innovative attention mechanism to capture the subtle changes in images. The mechanism uses attentional spatial pooling from a self-attention layer and learns the importance over fine-grained to coarse spatial structures and combine them to capture rich semantic information concerning a given rotation axis. The experimental evaluation of our approach using three benchmark datasets is very competitive to state-of-the-art methods, including with and without landmark-based approaches.

![Image](model1.jpg)
**RAFA-Net for estimating head poses by introducing rotation axis-specific (yaw, pitch and roll) self-attention and attentional pooling components.**

![Image](model.jpg)
**Learning pixel-level relationships from the convolutional feature map of size _W x H x C_. b) CAP using integral regions to capture both self and neighborhood contextual information. c) Encapsulating spatial structure of the integral regions using an LSTM. d) Classification by learnable aggregation of hidden states of the LSTM.**

### Paper and Supplementary Information
Extended version of the accepted paper in [ArXiv](https://arxiv.org/abs/2101.06635).

[Supplementary Document](AAAI_Supplementary.pdf)

[Source code](https://github.com/ArdhenduBehera/cap)

### Bibtex
```markdown
@inproceedings{behera2021context,
  title={Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification},
  author={Behera, Ardhendu and Wharton, Zachary and Hewage, Pradeep and Bera, Asish},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021},
  organization={AAAI}
}
```

### Acknowledgements

This research was supported by the UKIERI (CHARM) under grant DST UKIERI-2018-19-10. The GPU is kindly donated by the NVIDIA Corporation.
