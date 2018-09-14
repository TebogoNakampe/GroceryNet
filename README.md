# GroceryNet
GROCERYNET AT THE EDGE

<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/GroceryNet/blob/master/grocerynet.png">
</p>

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

Model Trained by https://github.com/cepiross/psucse_grocerycaffenet

A trend of classification accuracy in testing validation dataset indicates that Grocery-CaffeNet introduces fast convergence even with just an epoch due to derivation of parameters from CaffeNet. In the meantime, we can ensure potential of over-fitting issue by monitoring a trend of trian loss penalty. From our experiments, 45,000 iterations are probed to test the extent of available classification in the current setting. Our model may identify 21 grocery item classes in 76.2% accuracy.

| name                   | caffemodel                                                                                               | license      | sha1                                     |
|:----------------------:|:--------------------------------------------------------------------------------------------------------:|:------------:|:----------------------------------------:|
| Grocery-CaffeNet model | [caffenet\_train\_iter\_45000.caffemodel](https://drive.google.com/open?id=0B0lt6MbaK2RCZWd0ZklTMmVGbjg) | unrestricted | e43cb843634aae054a2a5bbb813967e0c63b5048 |

GroceryNet at the Edge
Using the Intel Movidius Neural Compute stick, and an off the shelf Raspberry Pi 3 and Pi Cam we
have created GroceryNet. Our AI is trained to detect grocery items and able to track the grocery items.
Intel® MovidiusTM VPUs drive the demanding workloads of GroceryNet at ultra-low power. By
coupling highly parallel programmable compute with workload-specific hardware acceleration, and co-
locating these components on a common intelligent memory fabric, Movidius achieves a unique
balance of power efficiency and high performance for GroceryNet.
We use Intel Movidius Neural Compute SDK tools for profiling, tuning, and compiling the fine tuned
CeffeNet deep neural network (DNN) model:
•
We used mvNCProfile command line tool to compileGroceryNet for use with the Intel®
MovidiusTM Neural Compute SDK (Intel® MovidiusTM NCSDK), runs the network on a
connected neural compute device, and outputs text and HTML profile reports.
The profiling data contains layer-by-layer statistics about the performance of the
grocerynetwork. This is helpful in determining how much time is spent on each layer to narrow
down potential changes to the network to improve the total inference time.


<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/GroceryNet/blob/master/model.png">
</p>


Replicate this project:

    cd workspace
    cd ncsdk
    cd examples
    cd caffe
    git clone git@github.com:TebogoNakampe/GroceryNet.git
    cd GroceryNet

download caffemodel into this folder

    make run
    
    
    

