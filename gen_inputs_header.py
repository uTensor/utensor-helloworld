import tensorflow as tf
import numpy as np
import argparse


def main(num_samples=5, seed=None):
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0

    # Add a channels dimension
    x_test = x_test[..., np.newaxis]
    total_N = y_test.shape[0]
    num_samples = min(num_samples, total_N)
    np.random.seed(seed)
    idxs = np.random.choice(range(total_N), num_samples, replace=False)
    x_selected = x_test[idxs].reshape(num_samples, -1)
    y_selected = y_test[idxs]
    with open("input_image.h", "w") as fid:
        fid.write("// clang-format off\n")
        fid.write(
            "const float arr_input_image[{}][{}] = {{\n".format(
                x_selected.shape[0], x_selected.shape[1]
            )
        )
        for i in range(x_selected.shape[0]):
            arr = x_selected[i]
            fid.write("    {{ {}".format(", ".join(map(str, arr))))
            fid.write("},\n")
        fid.write("};\n")
        fid.write("const int ref_labels[{}] = {{\n".format(y_selected.shape[0]))
        fid.write("    " + ", ".join(map(str, y_selected)) + "\n")
        fid.write("};\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        default=5,
        help="the number of inpute samples [default: %(default)s]",
        type=int,
        metavar="INTEGER",
    )
    parser.add_argument("--seed", default=None, help="the random seed", type=int)
    args = vars(parser.parse_args())
    main(**args)
