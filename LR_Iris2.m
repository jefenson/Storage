%�����߼��ع��㷨��Ŀ�����������
clear;
load iris_dataset     %����iris���ݼ� 
data = irisInputs';   %�ܹ�150*4ά���ݣ�������
setosa_Train = data(1:25,:);    %ɽ�β��ѵ����
setosa_Test = data(26:50,:);    %ɽ�β�����Լ�
versicolor_Train = data(51:75,:);    %��ɫ�β��ѵ����
versicolor_Test = data(76:100,:);    %��ɫ�β�����Լ�
virginica_Train = data(101:125,:);   %ά�������β��ѵ����
virginica_Test = data(126:150,:);    %ά�������β�����Լ�
train_Data{1} = [setosa_Train;versicolor_Train];      %ɽ�β�ͱ�ɫ�βѵ����
train_Data{2} = [versicolor_Train;virginica_Train];   %��ɫ�β��ά�������βѵ����
train_Data{3} = [virginica_Train;setosa_Train];       %ά�������β��ɽ�βѵ����
test_Data = [setosa_Test;versicolor_Test;virginica_Train];   %���Լ�����
train_Class = [zeros(25,1);ones(25,1)];         %ѵ�������
test_Class = zeros(75,1);   %���Լ����
label = [ones(25,1);2*ones(25,1);3*ones(25,1)];
theta{1} = [0;0;0;0];    %��ʼ��Ȩ��
theta{2} = [0;0;0;0];
theta{3} = [0;0;0;0];
beta = [0;0;0];          %��ʼ��ƫ��    
alpha = 0.003;        %ѧϰ��ȡ0.003
N = [2;2;2];          %��������
J{1}(1) = 0;             %��ʧ����
J{2}(1) = 0; 
J{3}(1) = 0; 
for index = 1:3
    for i = 1:50
        z = train_Data{index}(i,:) * theta{index} + beta(index);
        phi = Sigmoid(z);
        J{index}(2) = J{index}(1) - train_Class(i) * log(phi) - (1 - train_Class(i)) * log(1 - phi);
    end
    while abs(J{index}(N(index)) - J{index}(N(index) - 1)) > 0.01    %�������������ʧ������С��0.01����ֹͣ����
        N(index) = N(index) + 1;
        dtheta = [0;0;0;0];
        dbeta = 0;
        for i = 1:50
            z = train_Data{index}(i,:) * theta{index} + beta(index);
            phi = Sigmoid(z);
            dtheta = dtheta + (train_Class(i) - phi) * train_Data{index}(i,:)';   %����theta�������ݶ�ֵ
            dbeta = dbeta + train_Class(i) - phi;   %����beta�������ݶ�ֵ
        end
        theta{index} = theta{index} + alpha * dtheta;  %����theta��beta��ֵ
        beta(index) = beta(index) + alpha * dbeta;   
        for i = 1:50
            z = train_Data{index}(i,:) * theta{index} + beta(index);
            phi = Sigmoid(z);
            J{index}(N(index)) = J{index}(N(index) - 1) - train_Class(i) * log(phi) - (1 - train_Class(i)) * log(1 - phi);   %���´��ۺ���ֵ
        end  
    end
end

figure;
subplot(4,1,1);
plot(1:N(1),J{1});   %����ѵ������ͼ
xlabel('��������');
ylabel('��ʧ����');
title('ɽ�β���ͱ�ɫ�β������ѵ��');
hold on
subplot(4,1,2);
plot(1:N(2),J{2});   %����ѵ������ͼ
xlabel('��������');
ylabel('��ʧ����');
title('��ɫ�β����ά�������β������ѵ��');
hold on
subplot(4,1,3);
plot(1:N(3),J{3});   %����ѵ������ͼ
xlabel('��������');
ylabel('��ʧ����');
title('ά�������β����ɽ�β������ѵ��');
hold on

%ʹ�ò��Լ�����
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
scatter(1:75,test_Class,30,label,'filled'); %���Ʋ��Լ���������ɢ��ͼ
xlabel('�����������');
ylabel('�������');
title('������������');
