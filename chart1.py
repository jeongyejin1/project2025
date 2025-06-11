import matplotlib.pyplot as plt

# 데이터 정의
epochs = [1, 2, 3, 4]
train_loss = [29770.2249, 0.3299, 0.2056, 0.1454]
train_accuracy = [0.8574, 0.9525, 0.9711, 0.9777]
val_accuracy = [0.8292, 0.9072, 0.8982, 0.9005]

# 차트 생성
plt.figure(figsize=(10, 6))

# Train Loss (왼쪽 y축)
plt.plot(epochs, train_loss, 'r--o', label='Train Loss')

# Accuracy (오른쪽 y축)
plt.twinx()
plt.plot(epochs, train_accuracy, 'b-o', label='Train Accuracy')
plt.plot(epochs, val_accuracy, 'g-o', label='Validation Accuracy')

# 레이블 및 타이틀
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.xticks(epochs)  # Epoch 간격을 1로 설정

# 보조 y축 라벨 (Train Loss용)
plt.gca().figure.axes[0].set_ylabel('Train Loss')

# 그래프 표시
plt.grid(True)
plt.show()
