# AMII TensorFlow Neural Network Library

This is a small library to facilitate neural network experiments in TensorFlow v1.2's Python3 interface, with an emphasis on enabling easy TensorBoard visualization and analysis. It is used and maintained by graduate students at the University of Alberta's Alberta Machine Intelligence Institute (AMII) group.

TensorFlow is a young, flexible machine learning library, but I found it difficult to get started compared to [Torch](http://torch.ch/). But the TensorFlow ecosystem has grown quickly and is used by the likes of Google Brain and DeepMind, so I thought I would give it a try.

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) is an amazing visualization tool and it makes me recommend TensorFlow over Torch and [PyTorch](http://pytorch.org/)<sup>[1](#visfootnote)</sup>. Unfortunately, its interface isn't well documented yet, and it's difficult to produce model performance graphs that directly compare different models and parameters. But after figuring out how TensorBoard expects data to be structured, it wasn't too difficult to write some convenient abstractions to allow multiple models to be trained and simultaneously evaluated on multiple criteria and data sets in a way that produces informative performance graphs. These abstractions also setup TensorBoard's computational graph visualizations, which are fantastic as well.

This library aims to be a lightweight wrapper to TensorFlow that facilitates neural network experiments that are readily visualized in a useful way by TensorBoard. It is not meant to hide so much detail that it's hard to port code when you want to switch to one of the heavier duty TensorFlow interfaces like [Keras](https://keras.io/), [Sonnet](https://github.com/deepmind/sonnet), or the higher-level interfaces coming out of TensorFlow's `contrib` library.


<a name="visfootnote">1</a>: Although I haven't looked much to see if visualization tools like TensorBoard are being developed for other libraries.


## Installation

Checkout the source and run `make install`. This will use `pip` to install the source as a Python library.

Unfortunately, TensorFlow is actually two separate libraries, the CPU-based `tensorflow` library and the GPU-based `tensorflow-gpu` library. Normally this wouldn't be a problem, except that the libraries are actually incompatible with each other. So there can only be one. You'll need to install the particular version for your system yourself. There is a comment in `setup.py` reminding those who look there that it is an implicit dependency, but I don't know how to enforce this programmatically.


## Usage

### Example

`exe/amii_tf_nn_mnist_example` is a complete example script that will:

1. Download [MNIST](http://yann.lecun.com/exdb/mnist/).
2. Create single and double layer neural network classifiers.
3. Train both networks with [Adam](https://arxiv.org/abs/1412.6980) optimizers.
4. Emit TensorFlow summary events during training and periodic evaluation on both cross-entropy loss, classification accuracy, and L2 error. This last one is present just to highlight how one would specify multiple evaluation criteria.

The example should run in just a minute or two, even on a CPU, because the number and size of the epochs are small. All data and results will be written to a `tmp` directory in your working directory. If you point TensorBoard at `tmp/amii_tf_nn_mnist_example_1` (i.e. running `tensorboard --logdir tmp/amii_tf_nn_mnist_example_1`), you can see the performance of your models as they trained, as well as network parameter summary statistics and the computational graph that was run.

The `tensorboard` command will start an HTML server on port `6006` and output a link for your browser. Following this link will show you the TensorBoard interface. The *Scalars* page shows performance graphs, the *Graphs* page shows the computational graph, and the *Distributions* and *Histograms* pages show network parameter and activation statistics.

Besides providing data to learn about TensorBoard, the example script also provides guidance in how to write your own experiments.

## Writing Your own Experiments

### Background

The *Scalars* page of TensorBoard uses the concepts of *name space*, *run*, and *variable* to organize its graphs.

A *name space* is represented as an element of collapsable accordion widget that contains the graphs in the middle of the page. The 'criteria' name space contains performance graphs on criteria, like 'accuracy' or 'xentropy', over the course of training.

**Each graph represents a TensorFlow variable**. A significant point of confusion for me was that I thought variables represented lines on a graph, which is not true.

A *run* is a model--data-set combination, like `AdamSingleLayerFeedForward-testing`. On the filesystem, it is just a directory filled with events associated to the `AdamSingleLayerFeedForward` model under the `testing` data set. The trick that plots the performance of all runs on a single performance graph is the sharing of the graph's variable.

For example, to add the next point for `AdamSingleLayerFeedForward-testing` to the `accuracy` graph, we execute the following procedure:

1. Evaluate `AdamSingleLayerFeedForward-testing` on the classification accuracy. We now have the accuracy as a realized number, not just a TensorFlow node.
2. Run the merged summary TensorFlow graph with the classification accuracy variable set to `AdamSingleLayerFeedForward-testing` realized accuracy. This will return an event summary that TensorBoard can read.
3. Write the event summary to the `AdamSingleLayerFeedForward-testing` event directory.

This is done for every model, with every criterion, on each evaluation checkpoint.


## Development

After checking out the repo, run `make install` to install this library and its dependencies. You can run `make test` to execute tests, or `make test-cov` to run tests and show coverage (yes, I know the coverage is embarrassing right now, this is certainly "research code").


### Tests

The library is setup with [pytest](https://docs.pytest.org/en/latest/) and adding tests is highly encouraged, particularly accompanying new features.


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/AmiiThinks/amii-tf-nn.


## TODO

- Release as a Python library that can be installed with `pip install amii-tf-nn` without cloning the source.
- Save and restore model checkpoints.
- Save training events so that TensorBoard can show them. Right now, only evaluation data is being saved. As long as evaluation checkpoints aren't too few, this may not be much of a problem.
- Integrate more with TensorBoard, like printing image and embedding data.
- Improve README documentation for writing experiments.
- Improve code documentation.
- Add tests.


## License

This library was created by [Dustin Morrill](http://dmorrill10.github.io/) and is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
