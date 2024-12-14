from collections import defaultdict
import random
import re

class NgramModel:
    def __init__(self, n):
        self.n = n  # The 'n' in n-gram
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        
    def preprocess_text(self, text):
        """Clean and tokenize text."""
        # Convert to lowercase and add sentence boundaries
        text = text.lower()
        text = f"<s> {text} </s>"
        # Split into words and remove extra whitespace
        tokens = re.findall(r'\b\w+\b|<s>|</s>', text)
        return tokens
    
    def train(self, text):
        """Train the model on input text."""
        tokens = self.preprocess_text(text)
        self.vocabulary.update(tokens)
        
        # Generate n-grams and count their frequencies
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_word = tokens[i+self.n-1]
            self.ngrams[context][next_word] += 1
    
    def generate_next_word(self, context):
        """Generate the next word given a context."""
        if tuple(context) not in self.ngrams:
            return random.choice(list(self.vocabulary))
        
        # Get probability distribution for next words
        total_count = sum(self.ngrams[tuple(context)].values())
        choices = list(self.ngrams[tuple(context)].items())
        words, counts = zip(*choices)
        probabilities = [count/total_count for count in counts]
        
        return random.choices(words, probabilities)[0]
    
    def generate_text(self, num_words=20):
        """Generate new text using the trained model."""
        # Start with sentence boundary
        context = ['<s>'] * (self.n-1)
        generated_text = []
        
        for _ in range(num_words):
            next_word = self.generate_next_word(context)
            if next_word == '</s>':
                break
            generated_text.append(next_word)
            # Update context
            context = context[1:] + [next_word]
        
        return ' '.join(generated_text)

# Example usage
training_text = """
The quick brown fox jumps over the lazy dog.
A quick brown dog jumps over the lazy fox.
The lazy fox sleeps under the tree.
"""

# Create and train a trigram model (n=3)
model = NgramModel(n=3)
model.train(training_text)

# Generate some text
print("Generated text:")
print(model.generate_text(num_words=15))

# Example of probability distribution for a specific context
context = ('the', 'lazy')
if tuple(context) in model.ngrams:
    print(f"\nProbability distribution for context '{' '.join(context)}':")
    total = sum(model.ngrams[tuple(context)].values())
    for word, count in model.ngrams[tuple(context)].items():
        prob = count/total
        print(f"{word}: {prob:.2f}")