clc
clear all

% Load Data
load fisheriris
inputs=[];
outputs=[];

N=max(size(species));
for i=1:N
    if strcmp(species(i),'setosa')
        inputs=[inputs;meas(i,1:2)];
        outputs=[outputs;1];
    elseif strcmp(species(i),'versicolor')
        inputs=[inputs;meas(i,1:2)];
        outputs=[outputs;-1];
    end
end
X = inputs;
Y = outputs;



precision=10^-5;
Cost=1000;
%Training/Fitting SVM
Ker = Ker_Linear(X,X);
N= size(X,1);
H= diag(Y)*Ker*diag(Y);
f= - ones(N,1);
Aeq=Y';
beq=0;
A=[];
b=[];
lb = zeros(N,1);
ub = repmat(Cost,N,1);
alpha=quadprog(H,f,A,b,Aeq,beq, lb, ub);



serial_num=(1:size(X,1))';
serial_sv=serial_num(alpha>precision&alpha<Cost);

temp_beta0=0;
for i=1:size(serial_sv,1)
    temp_beta0=temp_beta0+Y(serial_sv(i));
    temp_beta0=temp_beta0-sum(alpha(serial_sv(i))*...
        Y(serial_sv(i))*Ker(serial_sv,serial_sv(i)));
end
beta0=temp_beta0/size(serial_sv,1);


%================================
% Plotting
%================================
figure
hold on
P = size(X,2);

if P ~=2
   warning('# of input X should be 2 for the 2D visualization!!')
end

plot(X(Y==1,1),X(Y==1,2),'ro',...
    'LineWidth', 4,...
    'MarkerSize', 4);

plot(X(Y==-1,1),X(Y==-1,2),'bs',...
    'LineWidth', 4,...
    'MarkerSize', 4);

%
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
scores = SVM_pred(xGrid, X, Y,alpha,beta0);

contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k',...
    'LineWidth', 4);

xlabel('$X_1$','FontSize', 18,...
    'Interpreter','latex');
ylabel('$X_2$', 'FontSize', 18,...
    'Interpreter','latex');
legend({'+1:setosa';'-1:versicolor'},'FontSize',16,'Location', 'Best');
hold off
% Maximize figure
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
str_fig = strcat('SVM_','_C=',num2str(Cost));
saveas(gcf, str_fig,'png');
saveas(gcf, str_fig);

%SVM_Pred
function Y_new = SVM_pred(X_new, X, Y,alpha,beta0)
% X is N*p
% X_new is new data with M*p, alpha is vector with N*1, beta0 is scalar

M = size(X_new,1);

Ker=Ker_Linear(X,X_new);


Y_new = sum(diag(alpha.*Y)*Ker,1)'+beta0*ones(M,1);

return
end

% Linear Kernel Function
function Y=Ker_Linear(X1,X2)
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=dot(X1(i,:),X2(j,:));
    end
end

return
end