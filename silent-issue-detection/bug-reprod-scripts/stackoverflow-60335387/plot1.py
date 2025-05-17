import json
import matplotlib.pyplot as plt

# Load data from JSON file
with open('config.json') as config_file:
    config = json.load(config_file)
with open(f"{config.get('output')}.json", 'r') as file:
    data = json.load(file)

# Extracting data
train_loss = data['train_loss']
train_accuracy = data['train_accuracy']
val_loss = data['validation_loss']
val_accuracy = data['validation_accuracy']
epochs = range(1, len(train_loss) + 1)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
plt.grid(True)
# Save the plot to a file (e.g., PNG format)
plt.savefig(f"{config.get('output')}.png")