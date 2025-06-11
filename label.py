import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("drugsComTest_filtered.csv")

# 전체 행 중 10%만 무작위로 선택 (즉, 10% 선택됨)
df_reduced = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

# 결과 확인 (선택)
print("원본 행 수:", len(df))
print("10% 선택 후 행 수:", len(df_reduced))

# 4400건을 제외한 원본 데이터 만들기
df_remaining = df.drop(df_reduced.index)

# 새로운 CSV 파일로 저장 (원본에서 4400건 제외한 데이터 저장)
df_remaining.to_csv("drugsComTest_filtered_remaining.csv", index=False)

# 선택된 4400건은 이미 df_reduced로 저장되었습니다.
df_reduced.to_csv("drugsComTest_filtered_4400.csv", index=False)
