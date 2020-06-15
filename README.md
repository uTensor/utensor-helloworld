# uTensor Hello World repository
This is a quick end-to-end, training-to-deployment, uTensor demo.
For this tutorial, we used an K64F board. However, any Mbed enabled board should work.

## Requirements
In a Python virtual environment, install the following:
- mbed-cli
- utensor-cgen

### Mbed-CLI Installation
The Arm cross-compiler is a dependency. It can be installed with Brew:
```
brew install arm-none-eabi-gcc
```
Install the Mbed-CLI
```
pip install mbed-cli
```

### utensor-cli Installation
Because we are tapping into the latest features in uTensor for this sample project, we will have to install the *utensor-cli* from source. Here's the instruction:
```
git clone https://github.com/uTensor/utensor_cgen.git
# if you prefer SSH: git@github.com:uTensor/utensor_cgen.git

# use the appropriate branch
cd utensor_cgen
git checkout re-arch-support

pip install -e .
```
## Clone the Sample Project
```
git clone git@github.com:uTensor/utensor-helloworld.git
# or, SSH: https://github.com/uTensor/utensor-helloworld.git
```

## Training and Code Generation
The sample project should already includes the generated code and is ready to be compiled. The [mnist_conv.ipynb](https://github.com/uTensor/utensor-helloworld/blob/re-arch-rc1/mnist_conv.ipynb) contains the instructors for training a convolutional neural network. It can be easily modified to suit your need.

You will need Jupyter-notebook and utensor-cli to be installed under the same Python virtual environment to run the code, from the project root:
```
jupyter-notebook mnist_conv.ipynb
```
Run through all the cells, the generated code and parameters are placed in the `models` and `constant` folders.

In the example notebook, there are two pieces of code that are specific to uTensor:
- Representative Dataset Generator
- uTensor one-liner export API

#### Representative Dataset Generator
Offline-quantization is a powerful way to reduce the memory requirement while running models on MCUs. It works by mapping 32-bit floating-point number to 8-bit fix-point representation thus reducing the memory footprint by about 4x.

Quantization are often applied in per-tensor-dimension basis, and it requires us to know the range of values we are working with. Estimating the ranges of model parameters is straight forward because they are typically constants.

The activation range, on the other hand, varies with the input values. For offline-quantization to work, we have to provide some samples of input data to the quantization routine, so it can estimate the activation ranges. The values of these activation ranges are then embedded into the generated code. The kernel accept these values and quantize the activation at the runtime.

The Python generator below provides input samples to the quantizaton routine.
 
```
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

```
tflm_keras_export(
    model,
    representive_dataset=representative_dataset_gen,
    model_name='my_model',
    target='utensor',
)
```

##### Rename the notebook

### Compile
```
$ mbed deploy
$ mbed compile -m auto -t GCC_ARM -f --sterm --baudrate=115200
```
Expected output:

```
Simple MNIST end-to-end uTensor cli example (device)
Predicted label: 7
```
