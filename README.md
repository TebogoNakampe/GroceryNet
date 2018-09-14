# GroceryNet
GROCERYNET AT THE EDGE

Introduction
According to World Health Organization, “Africa’s health depends on improved nutrition
a profound shift from communicable to noncommunicable diseases (NCDs) is under way in many parts
of the African Region. Globally, NCDs are estimated to kill 38 million people each year and they
threaten progress towards the UN Millennium Development Goals and influence the post-2015
development agenda. The four main types of NCDs are cardiovascular diseases (like heart attacks and
stroke), cancers, chronic respiratory diseases (such as chronic obstructed pulmonary disease and
asthma) and diabetes.”
Automatic food recognition is emerging as an important topic due to the increasing demand for better
dietary and nutritional assessment tools. We propose the use of GroceryNet as an assessment tool for
grocery detection, in order to speed up the process of analyzing diet and nutrition.
Real-time grocery detection using GroceryNet. GroceryNet is a fine tuned CaffeNet model, which is a
replication of AlexNet. CaffeNet has a slight computational advantage to AlexNet. The Max-Pooling
precedes the Local Response Normalization (LRN) so that LRN takes less compute and memory. We
call the fine-tuned model GroceryNet, since it exploits CaffeNet in order to recognize grocery items.
Training GroceryNet is based on ImageNet Dataset. For Inference at the edge we integrate Intel
Movidius VPUs to drive the demanding workloads of GroceryNet at ultra-low power. By coupling
highly parallel programmable compute with workload-specific hardware acceleration, and co-locating
these components on a common intelligent memory fabric, Movidius achieves a unique balance of
power efficiency and high performance for GroceryNet.

