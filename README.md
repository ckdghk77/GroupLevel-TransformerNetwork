# GroupLevel-TransformerNetwork

This repository contains the official PyTorch implementation of GL-TN.



*****

### Requirements
* Python 3.6 (only tested on 3.6)
* Pytorch 1.3.1 (only tested on 1.3.1)
* torchvision 0.4.2 (only tested on 0.4.2)

To install requirements with pip:
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

*****

### Training
* exp-type=2 (MNIST)
* exp-type=3 (Omniglot)
* exp-type=4 (Fashion MNIST)
* exp-type=5 (CIFAR-10)
* exp-type=6 (Custom data)  (e.g. KDEF, FEI, custom data)
* exp-type=7 (Custom data-heavy)  (e.g. Pokémon, custom data)

Custom data : You have to put your own data to directory(/dataset/custom).</br>
Custom data with transparent background: use --transparent=True (e.g. Pokémon)

Other data  : Automatically downloaded by torch-vision.

To train the model in the paper, run this command (prepare your own data for "custom option"):
<pre>
<code>
python proposed_train.py --data-num=5 --exp-type=2 --target-class=3 --seed=15 //MNIST experiment
python proposed_train.py --data-num=5 --exp-type=7 --is-rgb=True --seed=15 //Custom RGB data experiment

</code>
</pre>


### Evaluation

To snythesize new data, run this command (parameter must be same to training):

<pre>
<code>
python proposed_inference.py --data-num=5 --exp-type=2 --target-class=3 --seed=15
</code>
</pre>

### Plot result

To plot the synthesized sample, run this command:

<pre>
<code>
python plot_interpol.py
</code>
</pre>




