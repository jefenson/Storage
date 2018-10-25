%运用逻辑回归算法对目标进行三分类
clear;
load iris_dataset     %输入iris数据集 
data = irisInputs';   %总共150*4维数据，即特征
setosa_Train = data(1:25,:);    %山鸢尾花训练集
setosa_Test = data(26:50,:);    %山鸢尾花测试集
versicolor_Train = data(51:75,:);    %变色鸢尾花训练集
versicolor_Test = data(76:100,:);    %变色鸢尾花测试集
virginica_Train = data(101:125,:);   %维吉尼亚鸢尾花训练集
virginica_Test = data(126:150,:);    %维吉尼亚鸢尾花测试集
train_Data{1} = [setosa_Train;versicolor_Train];      %山鸢尾和变色鸢尾训练集
train_Data{2} = [versicolor_Train;virginica_Train];   %变色鸢尾和维吉尼亚鸢尾训练集
train_Data{3} = [virginica_Train;setosa_Train];       %维吉尼亚鸢尾和山鸢尾训练集
test_Data = [setosa_Test;versicolor_Test;virginica_Train];   %测试集数据
train_Class = [zeros(25,1);ones(25,1)];         %训练集类别
test_Class = zeros(75,1);   %测试集类别
label = [ones(25,1);2*ones(25,1);3*ones(25,1)];
theta{1} = [0;0;0;0];    %初始化权重
theta{2} = [0;0;0;0];
theta{3} = [0;0;0;0];
beta = [0;0;0];          %初始化偏置    
alpha = 0.003;        %学习率取0.003
N = [2;2;2];          %迭代次数
J{1}(1) = 0;             %损失函数
J{2}(1) = 0; 
J{3}(1) = 0; 
for index = 1:3
    for i = 1:50
        z = train_Data{index}(i,:) * theta{index} + beta(index);
        phi = Sigmoid(z);
        J{index}(2) = J{index}(1) - train_Class(i) * log(phi) - (1 - train_Class(i)) * log(1 - phi);
    end
    while abs(J{index}(N(index)) - J{index}(N(index) - 1)) > 0.01    %如果相邻两次损失函数差小于0.01，则停止迭代
        N(index) = N(index) + 1;
        dtheta = [0;0;0;0];
        dbeta = 0;
        for i = 1:50
            z = train_Data{index}(i,:) * theta{index} + beta(index);
            phi = Sigmoid(z);
            dtheta = dtheta + (train_Class(i) - phi) * train_Data{index}(i,:)';   %计算theta变量的梯度值
            dbeta = dbeta + train_Class(i) - phi;   %计算beta变量的梯度值
        end
        theta{index} = theta{index} + alpha * dtheta;  %更新theta和beta的值
        beta(index) = beta(index) + alpha * dbeta;   
        for i = 1:50
            z = train_Data{index}(i,:) * theta{index} + beta(index);
            phi = Sigmoid(z);
            J{index}(N(index)) = J{index}(N(index) - 1) - train_Class(i) * log(phi) - (1 - train_Class(i)) * log(1 - phi);   %更新代价函数值
        end  
    end
end

figure;
subplot(4,1,1);
plot(1:N(1),J{1});   %绘制训练过程图
xlabel('迭代次数');
ylabel('损失函数');
title('山鸢尾花和变色鸢尾花分类训练');
hold on
subplot(4,1,2);
plot(1:N(2),J{2});   %绘制训练过程图
xlabel('迭代次数');
ylabel('损失函数');
title('变色鸢尾花和维吉尼亚鸢尾花分类训练');
hold on
subplot(4,1,3);
plot(1:N(3),J{3});   %绘制训练过程图
xlabel('迭代次数');
ylabel('损失函数');
title('维吉尼亚鸢尾花和山鸢尾花分类训练');
hold on

%使用测试集检验
for i = 1:75
    z1 = test_Data(i,:) * theta{1} + beta(1);
    z2 = test_Data(i,:) * theta{2} + beta(2);
    z3 = test_Data(i,:) * theta{3} + beta(3);
    
    if z1 < 0 && z3 > 0
        test_Class(i) = 1;
    end
    
    if z1 > 0 && z2 < 0
        test_Class(i) = 2;
    end
    
    if z2 >0 && z3 < 0
        test_Class(i) = 3;
    end
end
subplot(4,1,4);
scatter(1:75,test_Class,30,label,'filled'); %绘制测试集分类结果的散点图
xlabel('测试样本序号');
ylabel('所属类别');
title('测试样本分类');
