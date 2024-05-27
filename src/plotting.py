import matplotlib.pyplot as plt

# Read the loss values from the file
loss_values = []
with open("validation_loss.txt", "r") as file:
    for line in file:
        loss_values.append(float(line.strip()))

# Generate a list of epoch numbers starting from 1
epochs = list(range(1, len(loss_values) + 1))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, linestyle='-', color='black', label='Validation Loss')
plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.legend()
plt.savefig("validation_loss_curve.png", dpi=300)
