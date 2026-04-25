from flask import Flask, request, jsonify, render_template
import my_lightgbm_predict as predict_house
import traceback
import os

app = Flask(__name__)

@app.route('/')
def index():
    """首页 - 简洁科技温馨风格"""
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    """预测页面 - 数据输入界面"""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """预测API接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        price = predict_house.predict_house_price(data)
        return jsonify({'price': price})
    except Exception as e:
        error_msg = f'预测失败: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)  # 后端日志
        return jsonify({'error': error_msg}), 500

# Vercel需要这个变量
app = app

if __name__ == '__main__':
    # 生产环境配置
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)