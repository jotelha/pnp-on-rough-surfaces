input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

csv_file = input.csv_file

ELBO_histogram_png = output.ELBO_histogram_png
minibatch_speedup_png = output.minibatch_speedup_png
predictions_before_training_png = output.predictions_before_training_png
convergence_png = output.convergence_png
predictions_after_training_png = output.predictions_after_training_png
X_txt = output.X_txt
predicted_Y_txt = output.predicted_Y_txt
predicted_variance_txt = output.predicted_variance_txt
model_txt = output.model_txt

length_scale = float(wildcards.length_scale)
signal_variance = float(wildcards.signal_variance)

long_length_scale = float(wildcards.long_length_scale)
long_signal_variance = float(wildcards.long_signal_variance)

number_of_species = config["number_of_species"]

y_value_label = f"excess_concentration_integral_{wildcards.species}"

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import time
import gpflow
import tensorflow as tf

df = pd.read_csv(csv_file)

# Adapted from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
# permalink https://github.com/GPflow/docs/blob/e8e9bf9d401f2776050846f3372f43cb1acc5f8c/doc/source/notebooks/advanced/gps_for_big_data.ipynb

nb_points = 1001

def plot(title=""):
    plt.figure(figsize=None)
    plt.title(title)
    pX = np.linspace(interval[0], interval[1], nb_points)[:, None]
    pY, pYv = m.predict_y(pX)
    plt.plot(X, Y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color=col,
        alpha=0.6,
        lw=1.5,
    )
    Z = m.inducing_variable.Z.numpy()
    plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    plt.legend(loc="lower right")


# for y_value_label, directory in zip(y_value_labels_l, directories):
x_values = df["x"]

logger.info(f"Number of data points: %d", len(x_values))

y_values = df[y_value_label]
# remove mean
y_mean = y_values.mean()
y_values -= y_mean
logger.info(f"Removed mean {y_mean}.")

y_values_T = y_values.values.reshape(-1, 1)
x_values_T = x_values.values.reshape(-1, 1)

# set kernel parameters
# length_scale_l = [1., 0.1]
# signal_variance_l = [10*np.var(y_values), np.var(y_values)]
# long_length_scale = 10*length_scale
# long_signal_variance = 10*signal_variance
interval = [np.min(df["x"]), np.max(df["x"])]
# noise_variance = np.var(y_values) * 0.01

logger.info(f"Length scale {length_scale}.")
logger.info(f"Signal variance {signal_variance}.")

logger.info(f"Long length scale {long_length_scale}.")
logger.info(f"Long signal variance {long_signal_variance}.")

# logger.info(f"Length scales {length_scale_l}.")
#logger.info(f"Signal variances {signal_variance_l}.")

# shuffle input values
data = np.hstack([x_values_T, y_values_T])
np.random.shuffle(data)
x_values, y_values = data.T
y_values_T = y_values.reshape(-1, 1)
x_values_T = x_values.reshape(-1, 1)

# model
logger.info(f"Initialize model.")
X = x_values_T
Y = y_values_T
data = (X, Y)

N = len(Y)
M = 50  # Number of inducing locations
Z = X[:M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

kernel = gpflow.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scale) \
       + gpflow.kernels.SquaredExponential(variance=long_signal_variance, lengthscales=long_length_scale)

# kernel = gpflow.kernels.SquaredExponential(variance=signal_variance, lengthscales=length_scale)

m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

# Stochastical estimation of ELBO
elbo = tf.function(m.elbo)
tensor_data = tuple(map(tf.convert_to_tensor, data))
elbo(tensor_data)  # run it once to trace & compile

minibatch_size = 100
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N).batch(minibatch_size)

# ELBO computation with minibatches
ground_truth = m.elbo((X, Y)).numpy()
evals = [m.elbo(next(iter(train_dataset))).numpy() for _ in range(100)]

plt.figure(figsize=None)
plt.hist(evals, label="Minibatch estimations")
plt.axvline(ground_truth, c="k", label="Ground truth")
plt.axvline(np.mean(evals), c="g", ls="--", label="Minibatch mean")
plt.legend()
plt.title("Histogram of ELBO evaluations using minibatches")
logger.info("Discrepancy between ground truth and minibatch estimate: %g", ground_truth - np.mean(evals))

plt.savefig(ELBO_histogram_png)

# Minibatches speed up computation

# Evaluate objective for different minibatch sizes
minibatch_proportions = np.logspace(-2, 0, 10)
times = []
objs = []
for mbp in minibatch_proportions:
    batchsize = int(N * mbp)
    train_iter = iter(train_dataset.batch(batchsize))
    start_time = time.time()
    objs.append([elbo(minibatch) for minibatch in itertools.islice(train_iter, 20)])
    times.append(time.time() - start_time)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=None)
ax1.plot(minibatch_proportions, times, "x-")
ax1.set_xlabel("Minibatch proportion")
ax1.set_ylabel("Time taken")

ax2.plot(minibatch_proportions, np.array(objs), "kx")
ax2.set_xlabel("Minibatch proportion")
ax2.set_ylabel("ELBO estimates")

f.savefig(minibatch_speedup_png)

plot(title="Predictions before training")
plt.savefig(predictions_before_training_png)

# Running stochastic optimization
minibatch_size = 100

# We turn off training for inducing point locations
# gpflow.set_trainable(m.inducing_variable, False)

maxiter = 5000

@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        loss = m.training_loss((X, Y))
    gradients = tape.gradient(loss, m.trainable_variables)
    optimizer.apply_gradients(zip(gradients, m.trainable_variables))

optimizer = tf.keras.optimizers.Adam()

# Running the optimization
logf = []
for step in range(maxiter):
    optimization_step()
    if step % 10 == 0:
        elbo = m.elbo((X, Y)).numpy()
        logf.append(elbo)

plt.plot(np.arange(maxiter)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")

plt.savefig(convergence_png)

plot("Predictions after training")
plt.savefig(predictions_after_training_png)

# dump predictions
pX = np.linspace(interval[0], interval[1], nb_points)[:, None]
pY, pYv = m.predict_y(pX)  # Predict Y values at test locations

np.savetxt(X_txt, pX.flatten())
np.savetxt(predicted_Y_txt, pY.numpy().flatten() + y_mean)
np.savetxt(predicted_variance_txt, pYv.numpy().flatten())

# dump model
gpflow.utilities.print_summary(m)
s = gpflow.utilities.tabulate_module_summary(m)
with open(model_txt,'w') as f:
   f.write(s)