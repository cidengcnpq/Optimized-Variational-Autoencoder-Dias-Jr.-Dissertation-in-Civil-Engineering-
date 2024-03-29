function [TrainData,ValidationData,TestData,OptimizationData] = CarregaSinais_Z24_Tempo(num_col,num_points,num_ac,n_tr,caminho)
%% 1)IMPORTANDO ARQUIVOS

%Defini��o do caminho para essa pasta

filePath = matlab.desktop.editor.getActiveFilename;
[pathstr,name,ext]  = fileparts(filePath);
parentDir=fileparts(pathstr);
cd(parentDir);
addpath(genpath(parentDir));

%Carregamento dos dados de treinamento e valida��o
%Dano 1
 numParts=52;
 TamDiv = 1260;
 data_D1_red = cell(numParts*num_col,1);
 aux=0;

for j = 1:num_col
        % Carregando matrizes de dados [2000(pontos eixo x) x 1(acelerometers)] 
        load([caminho,'\xc1_',mat2str(j),'.mat']);
        data_D1{j,1}=X(1:num_points,num_ac);
              
            for i = 1:numParts
                startIndex = (i - 1) * TamDiv + 1;
                endIndex = i * TamDiv;
                data_D1_red{i+aux,1} = data_D1{j}(startIndex:endIndex);
            end
    aux=aux+numParts;
end


%Dano 2
aux=0;
for j = 1:num_col
        % Carregando matrizes de dados [2000(pontos eixo x) x 1(acelerometers)] 
        load([caminho,'\xc2_',mat2str(j),'.mat']);
        data_D2{j,1}=X(1:num_points,num_ac);
        
               for i = 1:numParts
                startIndex = (i - 1) * TamDiv + 1;
                endIndex = i * TamDiv;
                data_D2_red{i+aux,1} = data_D2{j}(startIndex:endIndex);
            end
        aux=aux+numParts;
end

%Dano 4
aux=0;
for j = 1:num_col
        % Carregando matrizes de dados [2000(pontos eixo x) x 1(acelerometers)] 
        load([caminho,'\xc4_',mat2str(j),'.mat']);
        data_D4{j,1}=X(1:num_points,num_ac);
        
               for i = 1:numParts
                startIndex = (i - 1) * TamDiv + 1;
                endIndex = i * TamDiv;
                data_D4_red{i+aux,1} = data_D4{j}(startIndex:endIndex);
            end
        aux=aux+numParts;
end


%Dano 5
aux=0;
for j = 1:num_col
        % Carregando matrizes de dados [2000(pontos eixo x) x 1(acelerometers)] 
        load([caminho,'\xc5_',mat2str(j),'.mat']);
        data_D5{j,1}=X(1:num_points,num_ac);
        
            for i = 1:numParts
                startIndex = (i - 1) * TamDiv + 1;
                endIndex = i * TamDiv;
                data_D5_red{i+aux,1} = data_D5{j}(startIndex:endIndex);
            end
        aux=aux+numParts;
end


data_D1=data_D1_red;
data_D2=data_D2_red;
data_D4=data_D4_red;
data_D5=data_D5_red;

clear data_D1_red data_D2_red data_D4_red data_D5_red 


% Colocando os dados entre [-1,1]

%Dados dano 1
for k=1:length(data_D1)
    x=data_D1{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D1{k,1}=norm;
end

%Dados dano 2
for k=1:length(data_D2)
    x=data_D2{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D2{k,1}=norm;
end

%Dados dano 4
for k=1:length(data_D4)
    x=data_D4{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D4{k,1}=norm;
end

%Dados dano 5
for k=1:length(data_D5)
    x=data_D5{k};
    maxAbsVal = max(abs(x(:,1)));
    norm=x./maxAbsVal;
    data_norm_D5{k,1}=norm;
end

%% 2)TREINO/TESTE/OTIMIZA��O
%Aleatorizando os dados para treino/valida��o/otimiza��o
data_ale_D1=data_norm_D1(randperm(length(data_norm_D1)));

%Dividindo os dados para treinamento
TrainData=[data_ale_D1(1:n_tr)];

%Dividindo os dados para valida��o
ValidationData=[data_ale_D1((n_tr+1):end)];

%Dividindo os dados para teste
TestData=[vertcat(data_norm_D2,data_norm_D4,data_norm_D5)];

%Dividindo os dados para otimiza��o
OptimizationData=TrainData;

end