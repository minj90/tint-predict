# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # 1. 사전 로딩된 LSTM 모델
# model = load_model('./new_model/29-0.7149.keras')


# # 2. 유저 입력 -> LSTM 입력 (3일치 * 8피처) 변환 함수
# def convert_user_input_to_model_input(orders):
#     df = pd.DataFrame(orders)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values(by='date')

#     latest_date = df['date'].max()
#     lookback_days = 3 
#     date_range = [latest_date - timedelta(days=i) for i in range(lookback_days -1,-1,-1)]

#     filled = []
#     for date in date_range :
#         row = df[df['date'] == date]
#         if row.empty :
#             filled.append({'date':date, "qty": 0})
#         else :
#             filled.append({'date':date, "qty": int(row['qty'].values[0])})
#     filled_df = pd.DataFrame(filled)
#     qtys = filled_df['qty'].values

#     features = []
#     for i in range(len(qtys)) :
#         qty = qtys[i]
#         mean_qty = np.mean(qtys[:i + 1]) if i>0 else qty
#         diff = qtys[i] - qtys[i-1] if i>0 else 0
#         cumulative = np.sum(qtys[:i +1])
#         features.append([
#             qty, #현재 수량
#             mean_qty, #평균 수량
#             diff, #증감량
#             cumulative, #누적 수량
#             1 if qty>0 else 0, # 발주 여부
#             i/ lookback_days, # 상대 시간 위치(0.0, 0.33, 0.66) # lookback_days=3
#             np.log1p(qty), #로그 스케일 수량(log(1+qty))
#             qty / (mean_qty + 1e-5) # 현재수량 / 평균(비율)
#         ])

#     return np.array(features).reshape(1,3,8)
    
#     # 예측 API 엔드 포인트
#     @app.route('/predict', methods=['POST'])
#     def predict():
#         try:
#             data = request.get_json()
#             #클라이언트가 보낸 JSON 데이터를 딕셔너리 형태로 파싱
#             #예 : {"orders": [{"data": "2025-04-09", "qty": 100},...]}
#             orders = data.get('orders')
#             # JSON에서 "orders" 키에 해당하는 값 (리스트 형태)를 가져온다
#             # 예 : [{"date":....,"qty":....},....]

#             # 입력 검증
#             if not orders:
#                 # orders 값이 없거나 비어 있다면 -> 400 Bad Request 와 함께 오류 메시지 반환
#                 return jsonify({'error' : 'No orders provided'}), 400
            
#             model_input = convert_user_input_to_model_input(orders)
#             y_pred = model.predict(model_input).flatten()[0]

#             return jsonify({'prediction' : float(y_pred)}) # 예측 결과를 JSON으로 감싸서 반환
        
#         # 예외처리
#         except Exception as e :
#             return jsonify({'error': str(e)}), 500
#         # 오류가 발생하면 HTTP 500 응답과 함께 에러 메시지를 반환

# # 4. 서버 실행
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#     #기본적으로 localhost:5000에서 실행됨, port=5000 생략가능
#     # 기본값은 host='127.0.0.1 -> 오직 보인 컴퓨터 (로컬)에서만 접속가능
#     # host='0.0.0.0' -> 같은 네트워크 내 다른 기기에서도 접속 가능
#     # 외부에서 접속하게 만들고 싶다면 꼭 host='0.0.0.0' 써야 한다.

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 1. 사전 로딩된 LSTM 모델
model = load_model('./new_model/29-0.7149.keras')

# 2. 유저 입력 → LSTM 입력 (3일치 × 8피처) 변환 함수
def convert_user_input_to_model_input(orders):
    df = pd.DataFrame(orders)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    latest_date = df['date'].max()
    lookback_days = 3
    date_range = [latest_date - timedelta(days=i) for i in range(lookback_days - 1, -1, -1)]

    filled = []
    for date in date_range:
        row = df[df['date'] == date]
        if row.empty:
            filled.append({"date": date, "qty": 0})
        else:
            filled.append({"date": date, "qty": int(row['qty'].values[0])})

    filled_df = pd.DataFrame(filled)
    qtys = filled_df['qty'].values

    features = []
    for i in range(len(qtys)):
        qty = qtys[i]
        mean_qty = np.mean(qtys[:i + 1]) if i > 0 else qty
        diff = qtys[i] - qtys[i - 1] if i > 0 else 0
        cumulative = np.sum(qtys[:i + 1])
        features.append([
            qty,
            mean_qty,
            diff,
            cumulative,
            1 if qty > 0 else 0,
            i / lookback_days,
            np.log1p(qty),
            qty / (mean_qty + 1e-5)
        ])

    return np.array(features).reshape(1, 3, 8)

# 3. 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
       data = request.get_json()
       #클라이언트가 보낸 JSON 데이터를 딕셔너리 형태로 파싱  
       #예: {"orders": [{"date": "2025-04-09", "qty": 100}, ...]}
       orders = data.get('orders')
       #JSON에서 "orders" 키에 해당하는 값(리스트 형태)을 가져온다. 
       # 예: [{"date": ..., "qty": ...}, ...]

       #입력 검증
       if not orders:
           # orders 값이 없거나 비어 있다면 → 400 Bad Request와 함께 오류 메시지 반환
           return jsonify({'error': 'No orders provided'}), 400
       
       model_input = convert_user_input_to_model_input(orders)
       y_pred = model.predict(model_input).flatten()[0]

       return jsonify({'prediction': float(y_pred)}) #예측 결과를 JSON으로 감싸서 반환
    
    #예외 처리
    except Exception as e:
       return jsonify({'error': str(e)}), 500
    #오류가 발생하면 HTTP 500 응답과 함께 에러 메시지를 반환 

# 4. 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# 기본적으로 localhost:5000 에서 실행됨, port=5000 생략가능
# 기본값은 host='127.0.0.1' → 오직 본인 컴퓨터(로컬)에서만 접속 가능
# host='0.0.0.0' → 같은 네트워크 내 다른 기기에서도 접속 가능
# 외부에서 접속하게 만들고 싶다면 꼭 host='0.0.0.0' 써야 한다.
