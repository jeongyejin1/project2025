# 필요한 라이브러리 다시 불러오기 및 데이터 재로드
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 데이터 로드
file_path = "C:\dev\project2025-main\drugsComTest_filtered_remaining.csv"
df = pd.read_csv(file_path)

# 날짜 컬럼 처리
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 오류값은 NaT로 처리

# 월 단위로 그룹화해서 리뷰 수 카운트
monthly_reviews = df.set_index('date').resample('M').size()

plt.rcParams['font.family'] = 'Malgun Gothic'
# 시각화
plt.figure(figsize=(14, 6))
plt.plot(monthly_reviews.index, monthly_reviews.values, color='teal', linewidth=2)
plt.title("시간에 따른 리뷰 수 변화", fontsize=16)
plt.xlabel("리뷰 날짜 (월 단위)", fontsize=12)
plt.ylabel("리뷰 수", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# x축 포맷 설정 (연-월 표시)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
