import jax.numpy as jnp
from jax import random, grad, vmap
import flax

# Define the model using Flax
class Model(flax.nn.Module):
    def apply(self, x):
        dense_layer = flax.nn.Dense(x, 1)
        return jnp.squeeze(jnp.where(dense_layer > 0.5, 1., 0.), -1)

# Initialize the model
_, params = Model.init(random.PRNGKey(0), x=jnp.zeros((1, 2)))

# Define the loss function and the accuracy function
def loss_fn(model, x, y):
    logits = model(x)
    return jnp.mean(jnp.square(logits - y))

def accuracy_fn(model, x, y):
    logits = model(x)
    return jnp.mean(jnp.equal(jnp.round(logits), y))

# Prepare the input and target data
x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = jnp.array([[0], [1], [1], [0]])

# Define the optimizer
opt_init, opt_update, get_params = flax.optim.sgd(0.1)
opt_state = opt_init(params)

# Train the model
for i in range(10000):
    opt_state = opt_update(i, grad(loss_fn)(Model, opt_state.params, x, y), opt_state)
    params = get_params(opt_state)

# Print the final output of the model
print(Model(params, x))
