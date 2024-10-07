import pandas as pd

# CSV 파일 경로 지정
csv_file_path = '/home/mean6021/hate_classification/prediction/result.csv'  # 파일 경로를 적절히 변경하세요

# CSV 파일 로드
data = pd.read_csv(csv_file_path)

# 데이터의 총 줄 수 확인
num_rows = len(data)
print(f"Total number of rows: {num_rows}")

# output 열의 값이 1인 행만 선택
output_1_rows = data[data['output'] == 1]

# 선택된 행의 개수 출력
print(f"Number of rows with output value 1: {len(output_1_rows)}")

# 1의 값만 있는 행 출력 (원하는 만큼 보여줄 수 있음)
print(output_1_rows)

