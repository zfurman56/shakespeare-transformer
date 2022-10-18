# shakespeare-transformer

Toy implementation of a decoder-only transformer for predicting Shakespeare text, using PyTorch. Run with `python shakespeare.py`.

Made this as a learning exercise for myself, but may be helpful for others to see. Reimplemented from (almost) scratch, though I allowed myself to look at reference implementations if I couldn't debug an issue within several hours.

## (Potentially obvious) lessons learned
- Debugging deep learning models is hard
- Do single-head attention first, then make it multi-head
- Nobody mentions how to calculate loss for a model like this
  - Flatten the output predictions from a batch, as well as the expected labels, as if you just had a single sequence of size `context_size*batch_size`. Your output predictions will be of size `(context_size*batch_size, vocab_size)`, and your labels will be of size `(context_size*batch_size)`. Then just use cross-entropy loss (in PyTorch, remember to use raw logits).
- Don't do in-place operations with PyTorch unless you know exactly what you're doing
  - It breaks autograd
  - PyTorch won't warn you
  - There's a difference between x += y and x = x + y
  - If your loss is increasing over time and you can't figure out why, this might be it
- Validate your attention mask with an experiment, it's probably wrong
  - For instance, feed tokens from a Markov model and make sure the trained loss matches the theoretical value
- You need an embedding layer, one-hot encoding is not compatible with positional encoding
  - Related: the model dimension / embedding dimension *is not* the same thing as vocab size
- Don't forget: for an autoregressive model, you need to shift the output labels to the left compared to the input labels
- Don't forget to add dropout
- All the things from http://karpathy.github.io/2019/04/25/recipe/
