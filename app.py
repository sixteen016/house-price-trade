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

def _adapt_frontend_data(data):
    """适配前端发送的数据（支持完整格式和简化格式）"""
    
    # 检查数据格式：如果是完整格式（包含'房屋户型'等字段），直接返回
    if '房屋户型' in data or '建筑面积' in data:
        print("检测到完整格式数据，直接使用")
        return data
    
    # 如果是简化格式，进行转换
    print("检测到简化格式数据，进行转换")
    
    # 前端简化格式到后端完整格式的映射
    mapping = {
        'area': '建筑面积',
        'room': '室',
        'hall': '厅',
        'ward': '卫',
        'kitchen': '厨',
        'floors': '总楼层',
        'floor': '实际楼层',
        'orientation': '房屋朝向',
        'decoration': '装修情况',
        'community': '小区名称'
    }
    
    adapted_data = {}
    
    # 处理直接映射的字段
    for frontend_key, backend_key in mapping.items():
        if frontend_key in data:
            if frontend_key == 'area':
                adapted_data[backend_key] = f"{data[frontend_key]}㎡"
            elif frontend_key in ['room', 'hall', 'ward', 'kitchen']:
                # 构建房屋户型字符串
                if '房屋户型' not in adapted_data:
                    adapted_data['房屋户型'] = f"{data.get('room', 2)}室{data.get('hall', 1)}厅{data.get('ward', 1)}卫"
                    if data.get('kitchen'):
                        adapted_data['房屋户型'] += f"{data['kitchen']}厨"
            elif frontend_key == 'floors':
                adapted_data[backend_key] = data[frontend_key]
                adapted_data['所在楼层'] = f"中楼层(共{data[frontend_key]}层)"
            elif frontend_key == 'floor':
                adapted_data[backend_key] = data[frontend_key]
            else:
                adapted_data[backend_key] = data[frontend_key]
    
    # 设置默认值
    adapted_data.setdefault('房屋户型', '2室1厅1卫')
    adapted_data.setdefault('建筑面积', '85.00㎡')
    adapted_data.setdefault('所在楼层', '中楼层(共18层)')
    adapted_data.setdefault('房屋朝向', '南')
    adapted_data.setdefault('装修情况', '精装')
    adapted_data.setdefault('建筑结构', '钢混')
    adapted_data.setdefault('建筑类型', '普通住宅')
    adapted_data.setdefault('产权性质', '商品房')
    adapted_data.setdefault('房权所属', '非共有')
    adapted_data.setdefault('配备电梯', '有')
    adapted_data.setdefault('梯户比例', '1梯2户')
    adapted_data.setdefault('建成年代', '2010')
    adapted_data.setdefault('挂牌时间', '2023-01-01')
    adapted_data.setdefault('成交周期（天）', 30)
    adapted_data.setdefault('调价（次）', 1)
    adapted_data.setdefault('带看（次）', 5)
    adapted_data.setdefault('关注（人）', 50)
    adapted_data.setdefault('浏览（次）', 200)
    
    return adapted_data

@app.route('/predict', methods=['POST'])
def predict():
    """预测API接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        # 适配前端数据格式
        adapted_data = _adapt_frontend_data(data)
        price = predict_house.predict_house_price(adapted_data)
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