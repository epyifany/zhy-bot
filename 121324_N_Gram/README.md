a simple n-gram language model implementation. Let's break down how it works:

The model uses a sliding window of n-1 words to predict the next word in a sequence
It learns by counting how often words appear after each context
When generating text, it samples from the learned probability distributions

Key components:

Text preprocessing (lowercase, adding sentence boundaries)
N-gram generation and counting
Probability-based text generation
Handling unseen contexts with backoff to random choice

To experiment with this code:

Try different values of n (2 for bigrams, 3 for trigrams, etc.)
Feed it different training texts
Adjust the generation length
Examine the probability distributions it learns
