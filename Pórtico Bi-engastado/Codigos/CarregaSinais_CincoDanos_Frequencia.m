function [TrainData,ValidationData,TestData,OptimizationData] = CarregaSinais_CincoDanos_Frequencia(hz,T,num_col,num_points,num_ac,n_tr,caminho)
%% 1)IMPORTANDO ARQUIVOS

%Definição do caminho para essa pasta

filePath = matlab.desktop.editor.getActiveFilename;
[pathstr,name,ext]  = fileparts(filePath);
parentDir=fileparts(pathstr);
cd(parentDir);
addpath(genpath(parentDir));

%Carregamento dos dados de treinamento e validação
%Dano 0

for j = 1:num_col
        load([caminho,'\CARGA 0\\carga_0_',mat2str(j),'.mat']);
        data_D0{j,1}=v(1:num_points,num_ac);
end

%Dano 1
for j = 1:num_col
        load([caminho,'\CARGA 1\\carga_1_',mat2str(j),'.mat']);
        data_D1{j,1}=v(1:num_points,num_ac);
end

%Dano 2
for j = 1:num_col
        load([caminho,'\CARGA 2\\carga_2_',mat2str(j),'.mat']);
        data_D2{j,1}=v(1:num_points,num_ac);
end

%Dano 3
for j = 1:num_col
        load([caminho,'\CARGA 3\\carga_3_',mat2str(j),'.mat']);
        data_D3{j,1}=v(1:num_points,num_ac);
end

%Dano 4
for j = 1:num_col
        load([caminho,'\CARGA 4\\carga_4_',mat2str(j),'.mat']);
        data_D4{j,1}=v(1:num_points,num_ac);
end


% Colocando os dados entre [-1,1]

%Dados dano 0
for k=1:num_col
    x=data_D0{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D0{k,1}=norm;
end

%Dados dano 1
for k=1:num_col
    x=data_D1{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D1{k,1}=norm;
end

%Dados dano 2
for k=1:num_col
    x=data_D2{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D2{k,1}=norm;
end

%Dados dano 3
for k=1:num_col
    x=data_D3{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D3{k,1}=norm;
end

%Dados dano 4
for k=1:num_col
    x=data_D4{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D4{k,1}=norm;
end

%% 1.1) Transformação para o domínio da frequência
% Discrtização de pontos
N=length(data_norm_D0{1});
% Intervalo de tempo entre 2 amostras (DT)
dt=T/N;
% Frequencia de Aquisicao
fs=N/T;
% Intervalo de frequencia entre 2 amostras (DT)
df=1/T;
% Eixo dos tempos
t=0:dt:(N-1)*dt;
% Eixo das frequencias
f=-N/2*df:df:(N-1)/2*df;
% Pontos no domínio da frequência
pts=hz/df+1;

%Dados dano 0
for k=1:num_col
    aux1=fft(data_norm_D0{k})/N;
    aux1s=abs(fftshift(aux1(1:pts,:)));
    aux2{k,1}=aux1s;
end
data_norm_D0=aux2;
clear aux2 aux1 aux1s


%Dados dano 1
for k=1:num_col
    aux1=fft(data_norm_D1{k})/N;
    aux1s=abs(fftshift(aux1(1:pts,:)));
    aux2{k,1}=aux1s;
end
data_norm_D1=aux2;
clear aux2 aux1 aux1s


%Dados dano 2
for k=1:num_col
    aux1=fft(data_norm_D2{k})/N;
    aux1s=abs(fftshift(aux1(1:pts,:)));
    aux2{k,1}=aux1s;
end
data_norm_D2=aux2;
clear aux2 aux1 aux1s

%Dados dano 3
for k=1:num_col
    aux1=fft(data_norm_D3{k})/N;
    aux1s=abs(fftshift(aux1(1:pts,:)));
    aux2{k,1}=aux1s;
end
data_norm_D3=aux2;
clear aux2 aux1 aux1s

%Dados dano 4
for k=1:num_col
    aux1=fft(data_norm_D4{k})/N;
    aux1s=abs(fftshift(aux1(1:pts,:)));
    aux2{k,1}=aux1s;
end
data_norm_D4=aux2;
clear aux2 aux1 aux1s


%% 2)TREINO/TESTE/OTIMIZAÇÃO
%Aleatorizando os dados para treino/validação/otimização
data_norm_D1234=vertcat(data_norm_D1,data_norm_D2,data_norm_D3,data_norm_D4); %União dos dados de dano D0 e D1
data_ale_D0=data_norm_D0(randperm(length(data_norm_D0)));

%Dividindo os dados para treinamento
TrainData=[data_ale_D0(1:n_tr)];

%Dividindo os dados para validação
ValidationData=[data_ale_D0((n_tr+1):end)];

%Dividindo os dados para teste
TestData=[data_norm_D1234];

%Dividindo os dados para otimização
OptimizationData=TrainData;

end