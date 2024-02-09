import json
import matplotlib.pyplot as plt

file = json.load(open('./models/earth_aug_4.json'))

print(file.keys())

train_loss = file['training_loss']
valid_loss = file['loss']
e = len(valid_loss)
cnt = int(len(train_loss) / e)

mean_train_loss = []
for i in range(0, len(train_loss), cnt):
    sum = 0
    for j in range(cnt):
        sum += train_loss[i + j]
    mean_train_loss.append(sum / cnt)

plt.plot(mean_train_loss, label="Train Set")
plt.plot(valid_loss, label="Valid Set")
plt.title("Train and Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()