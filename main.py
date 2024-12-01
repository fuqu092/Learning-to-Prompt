import torch
from dataset import get_dataloaders
from model import Model
from loss import CustomLoss

INPUT_SIZE = 224
NUM_CLASSES = 100
NUM_TASKS = 10
CLASSES_PER_TASK = 10
BATCH_SIZE = 16
EPOCHS_PER_TASK = 5
LEARNING_RATE = 0.001875
POOL_SIZE = 10
PROMPT_DIM = 5
TOP_K = 5
CONSTANT = 0.1
MAX_NORM = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloaders_train, dataloaders_test, class_mask = get_dataloaders(INPUT_SIZE, NUM_CLASSES, NUM_TASKS, CLASSES_PER_TASK, BATCH_SIZE, False, device)
model = Model(POOL_SIZE, PROMPT_DIM, TOP_K, NUM_CLASSES).to(device)
criterion = CustomLoss(CONSTANT).to(device)

print('TRAINING MODEL!!!')

model.train(True)

for i in range(NUM_TASKS):
    dataloader = dataloaders_train[i]
    mask = class_mask[i]
    total_task = 0.0
    correct_task = 0.0

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for j in range(EPOCHS_PER_TASK):
        total_epoch = 0.0
        correct_epoch = 0.0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            logits, loss_value = model(inputs)
            loss_value = loss_value.to(device)

            not_mask = np.setdiff1d(np.arange(NUM_CLASSES), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, labels, loss_value)
            optim.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            loss.backward()
            optim.step()

            _, predicted = torch.max(logits.data, 1)

            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            print(f'For Task{i+1} Epoch {j+1}/{EPOCHS_PER_TASK}, Accuracy: {correct_epoch/total_epoch * 100:.2f}%')

print('TESTING MODEL!!!')

model.eval()
with torch.no_grad():
    total_total = 0.0
    correct_total = 0.0

    for i in range(NUM_TASKS):
        dataloader = dataloaders_test[i]
        total_task = 0.0
        correct_task = 0.0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, distance_sum = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_task += labels.size(0)
            correct_task += (predicted == labels).sum().item()

        
        print(f'For Task{i+1} Accuracy: {correct_task/total_task * 100:.2f}%')
        
        total_total += total_task
        correct_total += correct_task

    print(f'Overall Accuracy: {correct_total/total_total * 100:.2f}%')