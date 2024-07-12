import matplotlib.pyplot as plt

# Example data
data = ["word1", "word2", "word3"]
text = " ".join(data)  # Join the words into a single string

plt.text(0.5, 0.5, text, verticalalignment='center', horizontalalignment='center')
plt.show()