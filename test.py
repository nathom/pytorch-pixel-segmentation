import torch

# Assuming you have a label batch tensor with shape (16, 224, 224)
# convert to (3,21,2,2)
label_batch = torch.randint(
    0, 4, (3, 2, 2), dtype=torch.uint8
).long()  # Replace this with your label tensor
# label_batch = label_batch.permute(1, 2, 0)
print(label_batch)

# Define the number of classes
num_classes = 4

# Apply one-hot encoding
one_hot_encoded = torch.nn.functional.one_hot(label_batch, num_classes)
print(one_hot_encoded,one_hot_encoded.shape)
one_hot_encoded = one_hot_encoded.permute(0,3,1,2)
print(one_hot_encoded)
# Print the shape of the resulting tensor
print(one_hot_encoded.shape)

batch1 = label_batch[0]
enc1 = one_hot_encoded[0]
