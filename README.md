# 房屋价格预测系统 (House Price Prediction System)

基于LightGBM机器学习算法的房屋价格预测Web应用，支持在线房价预测和数据分析。

## 项目特性

- 🏠 基于LightGBM的房价预测模型
- 🌐 Flask Web框架构建的RESTful API
- 📊 支持多种房屋特征输入
- 🚀 支持Vercel一键部署
- 📱 响应式Web界面

## 技术栈

- **后端**: Python, Flask, LightGBM, Scikit-learn
- **前端**: HTML, CSS, JavaScript
- **部署**: Vercel (无服务器部署)
- **数据处理**: Pandas, NumPy, Joblib

## 快速开始

### 环境要求

- Python 3.8+
- pip包管理器

### 安装依赖

```bash
pip install -r requirements.txt
```

### 本地运行

```bash
python app.py
```

访问 http://localhost:5000 查看应用

### API使用

```bash
POST /predict
Content-Type: application/json

{
  "area": 100,
  "room": 3,
  "hall": 2,
  "ward": 2,
  "kitchen": 1,
  "floors": 10,
  "floor": 5,
  "orientation": "南",
  "decoration": "精装",
  "community": "某某小区"
}
```

## 部署到Vercel

1. Fork此仓库
2. 在Vercel中导入项目
3. 配置环境变量（如果需要）
4. 部署完成

## 项目结构

```
├── app.py                 # Flask主应用
├── my_lightgbm_predict.py # 预测模型模块
├── requirements.txt       # Python依赖
├── vercel.json           # Vercel部署配置
├── templates/            # HTML模板
│   ├── index.html        # 首页
│   └── predict.html      # 预测页面
└── *.pkl                 # 模型文件
```

## 注意事项

- 模型文件较大，建议使用Git LFS管理
- Vercel部署时注意LightGBM的系统依赖问题
- 生产环境建议使用环境变量配置敏感信息

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！