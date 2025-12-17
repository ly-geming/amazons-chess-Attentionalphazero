# 贡献指南

感谢您对本项目的关注！欢迎提交 Issue 和 Pull Request。

## 开发环境设置

1. Fork 本仓库
2. 克隆您的 Fork：
```bash
git clone https://github.com/your-username/V10.git
cd V10
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 编译 C++ 扩展：
```bash
python setup.py build_ext --inplace
```

## 提交代码

1. 创建功能分支：
```bash
git checkout -b feature/your-feature-name
```

2. 提交更改：
```bash
git add .
git commit -m "描述您的更改"
```

3. 推送到您的 Fork：
```bash
git push origin feature/your-feature-name
```

4. 在 GitHub 上创建 Pull Request

## 代码规范

- 遵循 PEP 8 Python 代码规范
- 使用有意义的变量和函数名
- 添加必要的注释和文档字符串
- 确保代码通过基本测试

## 报告问题

在提交 Issue 时，请包含：
- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（Python 版本、操作系统等）

## 功能请求

欢迎提出新功能建议！请在 Issue 中详细描述：
- 功能用途
- 实现思路（可选）
- 可能的替代方案

