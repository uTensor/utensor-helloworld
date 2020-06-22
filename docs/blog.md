# TinyML via Code Generation

 In an earlier post, I discussed some motivations of why we created uTensor to bring ML to MCUs. There are currently three ways to deploy ML models to MCUs: Interpreter, Code-Generation, and Compiler. They each have their trade-offs as summarized in the table below:

 - Memory Usage
 - Code Size
 - Speed
 - Hackability
 - Workflow?

  uTensor employs a code-generation approach where C++ code is generated from a trained model. The generated code can be copied-and-pasted into embedded projects for easy integration, as shown in the illustration below:

  [flow graph]
  
  The code size contributes to uTensor's core is less than 2kB. It will support both online and offline memory planning. It integrates well with optimized computational kernels, for example, CMSIS-NN. Finally, its toolchain is written in pure Python enables one to prototype ideas with ease.

  We find the code-generation approach with a super customizable toolchain is the sweet spot of all TinyML aspects mentioned above. The rest of the tutorial presents the steps to set up your environment and deploy your first model with uTensor.

%mbed blog: old features of uTensor + new additions

## Requirements
- Brew (for MacOS)
- Python
- uTensor-CLI
- Jupyter
- Git & Mercurial
- Mbed-CLI
- ST-Link *(optional, only for ST boards)*


## Environment Setup
  We showcase the instructions for MacOS here; however, similar steps apply to other systems as well. Brew is a user-space package manager for MacOS. It is installed with a one-liner:

### Install Brew
```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```
Other systems may use different package managers, for example, `apt` on Ubuntu Linux.

### Install and Setup Python
We should not use the system's Python for our work. On top of that, to keep Python dependencies manageable, we should create a Python virtual environment for TinyML developments.
#### Install [`pyenv`](https://github.com/pyenv/pyenv)
  `pyenv` is a nice little package that helps us to install and switch between different versions of Python runtime on our systems. It is installed with:
  ```bash
  $ brew update
  $ brew install pyenv
  
  # Add it to your shell (ZSH in this case)
  $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
  $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
  $ echo 'if command -v pyenv 1>/dev/null 2>&1; then  eval "$(pyenv init -)"; fi' >> ~/.zshrc
  ```
  Show all Python versions avaliable to us:
  ```bash
  $ pyenv install --list
  ```
  Let's install `Python 3.6.8` and set it to our default Python:
  ```bash
  $ pyenv install 3.6.8
  $ pyenv global 3.6.8
  $ python --version
Python 3.6.8
  ```
  Create a Python Virtual Environment for TinyML
  ```bash
  $ mkdir ~/.pyvenv
  $ python -m venv ~~/.pyvenv/ut
  # Add it to shell
  $ alias ut="source ~/.pyvenv/ut/bin/activate"
  $ source ~/.pyvenv/ut/bin/activate
  # Activate it
  $ ut
  (ut) $
  ```
### Install uTensor-CLI and Jupyter
We will install uTensor-CLI and Jupyter-Notebook to the `ut` virtual environment. Please note that we do not have to explicitly install TensorFlow here as it comes with the installation of uTensor-CLI.
```bash
(ut) $ pip install utensor_cgen jupyter
```
### Mbed-CLI Installation
Most of the [Mbed-CLI](https://os.mbed.com/docs/mbed-os/v6.0/quick-start/build-with-mbed-cli.html) dependencies are installed with Brew.
```bash
 (ut) $ brew install git mercurial 
 (ut) $ brew tap ArmMbed/homebrew-formulae
 (ut) $ brew install arm-none-eabi-gcc
```
Install the Mbed-CLI with Brew
```bash
(ut) $ pip install mbed-cli
```

## The Sample Project
Clone the [hello-world sample project](https://github.com/uTensor/utensor-helloworld) with `git`:
```bash
(ut) $ git clone https://github.com/uTensor/utensor-helloworld
(ut) $ git cd utensor-helloworld
```
Here's the content of the repository:
```
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── constants
│   └── my_model
│       └── params_my_model.hpp
├── input_image.h
├── main.cpp
├── mbed-os.lib
├── mbed_app.json
├── mnist_conv.ipynb
├── models
│   └── my_model
│       ├── my_model.cpp
│       └── my_model.hpp
└── uTensor.lib
```
The Jupyter-notebook, `mnist_conv.ipynb`, contains the training code and will invoke an uTensor API, which generates C++ code from the trained model. For simplicity's sake, the project already contains the generated C++ code in the `constant` and `models` folders, so they are ready to be compiled.

In the next section, we will walk through the code in `mnist_conv.ipynb`. The `constant` and `models` folders contain the model's parameters and architecture, respectively. Running the notebook will overwrite them.


## Model Creation and Code Generation
The Jupyter-notebook can be launched from the project root:
```bash
(ut) $ jupyter-notebook mnist_conv.ipynb &
```
The above command should open a browser window of the notebook. Run the notebook by selecting `Kernel` > `Restart & Run All` from its dropdown menu:
[img]

We will only include the code relating to the model architecture and uTensor in this article. The full notebook can be viewed online [here](https://github.com/uTensor/utensor-helloworld/blob/master/mnist_conv.ipynb), and it is easily modifiable to suit your application.

### Model Creation

#### Defining the Model
We defined a convulutional neural network with less than 5kB of parameters (with quantization):
```python
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(8, 3, activation='relu')
    self.pool = MaxPooling2D()
    self.flatten = Flatten()
    self.d1 = Dense(16, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x0 = self.pool(x)
    x1 = self.conv1(x0)
    x2 = self.pool(x1)
    x3 = self.flatten(x2)
    x4 = self.d1(x3)
    return self.d2(x4)

model = MyModel()
```
This is a good starting point for 2D datasets, though, you may consider removing the first pooling layer for your custom applications:

```python
x0 = self.pool(x)
```
 As you tweak the model architecture, pay attention not only to the accuracy of the model, but also the model parameter size.

#### Model Size and RAM Estimation
Oftentimes, the model's size and RAM usage can be estimated from the model architecture itself. Here's a helper function that prints the model parameters given the input shape:
```python
def print_model_summary(model, input_shape):
  input_shape_nobatch = input_shape[1:]
  model.build(input_shape)
  inputs = tf.keras.Input(shape=input_shape_nobatch)
  
  if not hasattr(model, 'call'):
      raise AttributeError("User should define 'call' method in sub-class model!")
  
  _ = model.call(inputs)
  model.summary()

#providing the model and its input shape
print_model_summary(model, x_test.shape)
```
Output:
```
Model: "my_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 12, 12, 8)         80        
_________________________________________________________________
max_pooling2d (MaxPooling2D) multiple                  0         
_________________________________________________________________
flatten (Flatten)            (None, 288)               0         
_________________________________________________________________
dense (Dense)                (None, 16)                4624      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                170       
=================================================================
Total params: 4,874
Trainable params: 4,874
Non-trainable params: 0
```
The total number of parameters is around 4,874. Because model parameters are typically constants from an inferencing perspective, they are stored in the ROM of your device.


Activations, on the other hand, vary with every inference cycle; they are placed in RAM. A simple way to estimate the RAM usage of your model is by looking at the `Output Shape` column. In a sequential model, a layer's output is treated as the input of the next, so they both have to be stored in RAM. We assume the memory manager or the offline-memory optimiser will re-use the memory for other layers, so we only have to look at one layer at a time.

In our case, the `MaxPooling2D` layer has an input of `12 * 12 * 8` and an output of `(6 * 6 * 8) == 228`. So, our estimate is `12 * 12 * 8 + 6 * 6 * 8 = 1440` as the minimum RAM required to run the model itself. Depend on the efficiency of the memory planning algorithm used, the actual RAM usage is often higher than the estimate.

#### Training
The training will run-through 15 epoch                                                                                                                    s. You should see output similar to this:
```
Epoch 1, Loss: 0.459749698638916, Accuracy: 86.40333557128906, Test Loss: 0.18603216111660004, Test Accuracy: 94.27000427246094
Epoch 2, Loss: 0.1707976907491684, Accuracy: 94.72833251953125, Test Loss: 0.13616280257701874, Test Accuracy: 95.6300048828125
                                .
                                .
                                .
Epoch 15, Loss: 0.06735269725322723, Accuracy: 97.87333679199219, Test Loss: 0.08755753189325333, Test Accuracy: 97.13999938964844
```
### Code Generation
#### Offline Quantization
Offline-quantization is a powerful way to reduce the memory requirement while running models on MCUs. It works by mapping 32-bit floating-point number to 8-bit fix-point representation, reducing the memory footprint by about 4x.

Quantization is often applied in a per-tensor-dimension basis, and it requires us to know the range of the dimension. A typical evaluation of a neural network layer consists of model parameters, inputs, and activations. Estimating the ranges of model parameters is straight forward because they are typically constants.

The activation range, on the other hand, varies with the input values. For offline-quantization to work, we have to provide some samples of input data to the quantization routine so that it can estimate the activation ranges. The values of these activation ranges are then embedded into the generated code. The kernels accept these values and quantize the activation at the runtime.

The Python generator below provides randomly sampled inputs to the quantization routine.
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
#### uTensor One-Liner Export API
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
The files under the `constant` and `models` folder should now be updated. Viewing the contents of the files will provide more intuition on how uTensor works.

## Device Code
The [main.cpp](https://github.com/uTensor/utensor-helloworld/blob/master/main.cpp) serves as a template on how to invoke a uTensor model.
### Initializing the Model
We first have to instantize the generated model:
```cpp
static My_model model;
```
The `My_model` class is generated from our trained model, and its name is set by the `model_name='my_model'` provided to the one-line export API mentioned in the previous section.

The `My_model` class is responsible for:
- [Operator registration](https://github.com/uTensor/utensor-helloworld/blob/0165cfe51b3ae08de26d0bb189f53942414abfe3/models/my_model/my_model.hpp#L8-L35)
- [Initalizing memory allocators with predetermined RAM usage](https://github.com/uTensor/utensor-helloworld/blob/0165cfe51b3ae08de26d0bb189f53942414abfe3/models/my_model/my_model.hpp#L33-L34)
- [Implementation of the model](https://github.com/uTensor/utensor-helloworld/blob/master/models/my_model/my_model.cpp)
### Working with Tensors
Once we have the model object, we need to create tensors to pass in the input and store the output. Here, we have two types of tensors:
- `RomTensor` contains a read-only pointer to data. It points to the input data, `arr_input_image`, defined in the [input_image.h](https://github.com/uTensor/utensor-helloworld/blob/0165cfe51b3ae08de26d0bb189f53942414abfe3/input_image.h#L2). 
- `RamTensor` represents data in RAM.

All tensor types require us to specify shapes and data types.
```cpp
  Tensor input_image = new RomTensor({1, 28, 28, 1}, flt, arr_input_image);
  Tensor logits = new RamTensor({1, 10}, flt);
```
The tensor values can be read and written by specifying the indexi and data type:
#### Read
```cpp
float pixel_value = static_cast<float>(input_image(0, 1, 1, 0));
```
#### Write
```cpp
input_image(0, 1, 1, 0) = static_cast<float>(1.234f);
```
### Running the Model
The code that invokes the model is:
```cpp
model.set_inputs({{My_model::input_0, input_image}})
    .set_outputs({{My_model::output_0, logits}})
    .eval();
```
The `set_inputs()` and `set_outputs()` methods bind the input and output tensors to the model respectively. The input and output names of the generated model can be seem in its [header file](https://github.com/uTensor/utensor-helloworld/blob/0165cfe51b3ae08de26d0bb189f53942414abfe3/models/my_model/my_model.hpp#L11-L12). In this case, they are `input_0` and `output_0` as shown in the code snippet above.

After invoking the `eval()` on the model, the output tensor, `logits` contains the inference result. As an example, an `argmax()` function can be implemented with:
```cpp
int argmax(const Tensor &logits) {
  uint32_t num_elems = logits->num_elems();
  float max_value = static_cast<float>(logits(0));
  int max_index = 0;
  for (int i = 1; i < num_elems; ++i) {
    float value = static_cast<float>(logits(i));
    if (value >= max_value) {
      max_value = value;
      max_index = i;
    }
  }
  return max_index;
}

int max_index = argmax(logits);
```

## Deployment
We will mostly use Mbed-CLI for this section. It provide a simple-to-use wrapper to our cross-compiler, package manager, flash, and serial communication functions.

The `.lib` files in the repository contains references to libraries on Github. Mbed-CLI uses these references to download the corresponding source code when it deploys the project.

### Compile and Flash
Connect your Mbed enabled board to your computer, then:
```bash
(ut) $ mbed deploy
(ut) $ mbed compile -m auto -t GCC_ARM -f --sterm
```
A brief summary of the arguments provided to `mbed compile`:
- `-m auto`: auto detect the board connected, and set it as the current target.
- `-t GCC_ARM`: use GCC as our cross-compiler toolchain. In this case, it is `arm-none-eabi-gcc` we installed using `brew` in an earlier step.
- `-f`: flash it to the device when compilation is completed
- `--sterm`: connect the device's serial terminal
This is the output you should see:
```bash
Link: utensor-helloworld
Elf2Bin: utensor-helloworld
| Module           |      .text |    .data |      .bss |
|------------------|------------|----------|-----------|
| [fill]           |    485(+0) |   13(+0) |  2450(+0) |
| [lib]/c.a        |  68199(+0) | 2574(+0) |   127(+0) |
| [lib]/gcc.a      |   7196(+0) |    0(+0) |     0(+0) |
| [lib]/m.a        |    340(+0) |    0(+0) |     0(+0) |
| [lib]/misc       |    180(+0) |    4(+0) |    28(+0) |
| [lib]/nosys.a    |     32(+0) |    0(+0) |     0(+0) |
| [lib]/stdc++.a   | 191922(+0) |  145(+0) |  5720(+0) |
| main.o           |  14696(+0) |    0(+0) |  5969(+0) |
| mbed-os/drivers  |    206(+0) |    0(+0) |     0(+0) |
| mbed-os/hal      |   1777(+0) |    8(+0) |   131(+0) |
| mbed-os/platform |   7679(+0) |  260(+0) |   357(+0) |
| mbed-os/rtos     |   7979(+0) |  168(+0) |  5972(+0) |
| mbed-os/targets  |   9221(+0) |   36(+0) |   386(+0) |
| models/my_model  |   8448(+0) |    0(+0) |     0(+0) |
| uTensor/src      |   6031(+0) |    0(+0) |    36(+0) |
| Subtotals        | 324391(+0) | 3208(+0) | 21176(+0) |
Total Static RAM memory (data + bss): 24384(+0) bytes
Total Flash memory (text + data): 327599(+0) bytes

Image: ./BUILD/K64F/GCC_ARM/utensor-helloworld.bin
--- Terminal on /dev/tty.usbmodem14102 - 9600,8,N,1 ---
Simple MNIST end-to-end uTensor cli example (device)
pred label: 7
```
Press `ctrl + c` to disconnect the serial terminal.

Notice code size of uTensor:
```
| uTensor/src      |   6031(+0) |    0(+0) |    36(+0) |
```
The uTensor core and all the operators required to run a convolutional neural network contribute to less than **6kB** in terms of binary size.
 
...