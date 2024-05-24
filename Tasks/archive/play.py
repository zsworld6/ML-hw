import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 准备训练数据的特征和目标变量
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置随机森林参数的网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 创建 GridSearchCV 对象
grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy')

# 训练模型
grid_rf.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters: {grid_rf.best_params_}")
print(f"Best cross-validation accuracy: {grid_rf.best_score_}")

# 在验证集上验证模型
y_pred = grid_rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# 准备测试数据（排除'id'字段）
X_test = test_data.drop('id', axis=1)

# 使用最佳模型预测测试数据的价格范围
test_predictions = grid_rf.predict(X_test)

# 将预测结果与测试数据ID结合
prediction_results = pd.DataFrame({
    'id': test_data['id'],
    'price_range': test_predictions
})

# 保存预测结果到 CSV 文件
prediction_results.to_csv('predicted_price_ranges_optimized.csv', index=False)
