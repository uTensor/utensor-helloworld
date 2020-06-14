# uTensor Hello World repository
This is a quick end-to-end, training-to-deployment, uTensor demo.

## Requirements
In a Python virtual environment, install the following:
- mbed-cli
- utensor-cgen

*Please see the installation guide for mbed-cli and utensor-cli setup (to be linked).*

### Training and Code Generation
Run `deep_mlp.ipynb` in Jupyter-notebook.
Running the notebook should replace the pre-generated files in the repository.

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
