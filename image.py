# 세션이 초기화되어 파일을 다시 불러와야 함
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt




# 파일 경로
file_path = "C:\dev\project2025-main\drugsComTest_filtered_remaining.csv"
df = pd.read_csv(file_path)

# 상위 10개 약품: usefulCount 기준
top_useful_drugs = df.groupby('drugName')['usefulCount'].sum().sort_values(ascending=False).head(10)

# 나눔 폰트가 없을 경우를 대비해 대체 폰트 사용
plt.rcParams['font.family'] = 'Malgun Gothic'

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=top_useful_drugs.values, y=top_useful_drugs.index, palette="crest")
plt.title("유용한 리뷰 수가 많은 약품 Top 10", fontsize=16)
plt.xlabel("유용한 리뷰 수 (총합)", fontsize=12)
plt.ylabel("약품명", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
