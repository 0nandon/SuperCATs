# SuperCATs
For more information, check out the paper on [paper link]. Also check out project page here [Project Page link]

> 
>In this work, we introduce a novel network, namely Su- perCATs, which aims to find a correspondence field be- tween visually similar images. SuperCATs stands on the shoulder of the recently proposed matching networks, Su- perGlue and CATs, taking the merits of both for construct- ing an integrative framework. Specifically, given key- points and corresponding descriptors, we first apply at- tentional aggregation consisting of self- and cross- graph neural network to obtain feature descriptors. Subse- quently, we construct a cost volume using the descriptors, which then undergoes a tranformer aggregator for cost aggregation. With this approach, we manage to replace the hand-crafted module based on solving an optimal transport problem initially included in SuperGlue with a transformer well known for its global receptive fields, making our approach more robust to severe deformations. We conduct experiments to demonstrate the effectiveness of the proposed method, and show that the proposed model is on par with SuperGlue for both indoor and out- door scenes.


# Network
Overview of our model is illustrated below:
![overview](fig/overview.png)
Structure of Transformer Aggregator is illustrated below:
![overview](fig/aggregator.png)
