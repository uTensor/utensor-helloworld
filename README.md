# uTensor Hello World Repository (Preview)
This is a quick end-to-end, training-to-deployment, uTensor tutorial.
The [K64F](https://os.mbed.com/platforms/FRDM-K64F/) is used in this tutorial. However, any Mbed enabled board should work.

This is a project **PREVIEW**. More updates and tutorials will be released. Please give us your feedback in the mean time.

## What's New with uTensor Re-Arch
- TF2 Support
- One-line Exporter API
- Improved Inference API
- Clear, Concise, and Debuggable Generated Code
- Deterministic Memory Usage
- Lower Static and Dynamic Memory Footprint
- TensorFlow Lite Micro Interoperability (shared kernels)
- Offline Quantization
- *Offline Memory Optimization **(WIP)***
- *PyTorch Importer **(WIP)***
- *Arduino Support **(WIP)***

## Requirements
In a Python virtual environment, install the following:
- [mbed-cli](https://os.mbed.com/docs/mbed-os/v6.0/build-tools/install-and-set-up.html)
- [utensor-cgen](https://github.com/uTensor/utensor_cgen)

### Mbed-CLI Installation
The Arm cross-compiler is a dependency. On MacOS, it can be installed with Brew:
```bash
$ brew install arm-none-eabi-gcc
```
Install the Mbed-CLI
```bash
$ pip install mbed-cli
```

### utensor-cli Installation
Because we are tapping into the latest features in uTensor for this sample project, we will have to install the *utensor-cli* from the source. Here's the instruction:
```bash
$ git clone https://github.com/uTensor/utensor_cgen.git
# if you prefer SSH: git@github.com:uTensor/utensor_cgen.git

# use the appropriate branch
$ cd utensor_cgen
$ git checkout re-arch-support
$ pip install -e .
```
## Clone the Sample Project
```bash
$ git clone https://github.com/uTensor/utensor-helloworld.git
# or, SSH: git@github.com:uTensor/utensor-helloworld.git
```

## Training and Code Generation
The sample project should already include the generated code and is ready to be compiled. The [mnist_conv.ipynb](https://github.com/uTensor/utensor-helloworld/blob/re-arch-rc1/mnist_conv.ipynb) contains the instructors for training a convolutional neural network. It can be easily modified to suit your need.

You will need Jupyter-notebook and utensor-cli to be installed under the same Python virtual environment to run the code, from the project root:
```bash
$ jupyter-notebook mnist_conv.ipynb
```
Run through all the cells, the generated code and parameters are placed in the `models` and `constant` folders.

In the example notebook, two pieces of code that are specific to uTensor:
- Representative Dataset Generator
- uTensor one-liner export API

#### Representative Dataset Generator
Offline-quantization is a powerful way to reduce the memory requirement while running models on MCUs. It works by mapping 32-bit floating-point number to 8-bit fix-point representation, thus reducing the memory footprint by about 4x.

Quantization is often applied in a per-tensor-dimension basis, and it requires us to know the range of the dimension. Estimating the ranges of model parameters is straight forward because they are typically constants.

The activation range, on the other hand, varies with the input values. For offline-quantization to work, we have to provide some samples of input data to the quantization routine so that it can estimate the activation ranges. The values of these activation ranges are then embedded into the generated code. The kernels accept these values and quantize the activation at the runtime.

The Python generator below provides input samples to the quantization routine.
 
```python
# representative data function
num_calibration_steps = 128
calibration_dtype = tf.float32

def representative_dataset_gen():
    for _ in range(num_calibration_steps):
        rand_idx = np.random.randint(0, x_test.shape[0]-1)
        sample = x_test[rand_idx]
        sample = sample[tf.newaxis, ...]
        sample = tf.cast(sample, dtype=calibration_dtype)
        yield [sample]
```

#### uTensor one-liner export API
With the trained model and its representative dataset generator, uTensor can generate the C++ implementation of the model by invoking:

```python
from utensor_cgen.api.export import tflm_keras_export

tflm_keras_export(
    model,
    representive_dataset=representative_dataset_gen,
    model_name='my_model',
    target='utensor',
)
```

### Compile
```bash
$ mbed deploy
$ mbed compile -m auto -t GCC_ARM -f --sterm --baudrate=115200
```
Expected output:

```
Simple MNIST end-to-end uTensor cli example (device)
Predicted label: 7
```

## Join Us
Thanks for checking out this sneak peek of the upcoming uTensor update! There are many ways you can get involved with the community:
### Star the Projects
[Starring the project](https://github.com/uTensor/uTensor) is a great way to recognize our work and support the community. Please help us to spread the words!
### Join us on Slack
Our [Slack workspace](https://join.slack.com/t/utensor/shared_invite/zt-6vf9jocy-lzk5Aw11Z8M9GPf_KS5I~Q) is full of discussions on the latest ideas and development in uTensor. If you have questions, ideas, or want to get involved in the project, Slack is a great place to start.