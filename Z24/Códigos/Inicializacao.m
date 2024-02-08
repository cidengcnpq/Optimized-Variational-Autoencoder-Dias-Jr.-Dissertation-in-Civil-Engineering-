%% 1) INICIALIZAÇÃO
clear, clc, close all

%Inserir o caminho até a pasta de ENSAIOS
caminho='C:\Users\Universidade Federal\Desktop\Luiz\Z24\Dados';

%Definição dos parâmetros inciais
n_tr=300; %Número de dados para treino (para cada estágio dano utilizado)
n_test=468; %Número de dados para teste (para cada estágio dano utilizado)
num_col = 9; % Número de colisões em cada estágio de dano
num_points=65530; %Número de pontos pegados por ensaio
T=10; %Tempo de aquisição do sinal (caso seja escolhido o domínio da frequência)
hz=63; %Tamanho em hz do sinal (caso seja domínio da frequência)
num_ac=1; %Número do acelerômetro analisado
N_latent=100; %Espaço latente

%Configuração da arquitetura da RNA
TipoAE='VAE'; %Tipo de autocodificador utilizado (pode ser 'VAE' ou 'AE')
CamadaCod={'Bi-LSTM'}; %Tipo de camada codificadora ('FC','LSTM' ou 'Bi-LSTM')
CamadaDec={'Bi-LSTM'}; %Tipo de camada codificadora ('FC','LSTM' ou 'Bi-LSTM')

%Intervalo de otimização para os parâmetros
NeuCod=[500 600]; %Intervalo de neurônios para otimização na camada codificadora
NeuDec= [500 600];%Intervalo de neurônios para otimização na camada decodificadora
NumEpoc=[1 100]; %Intervalo de épocas a serem otimizadas
ger=10; %Número de gerações para otimizar os parâmetros do AE
pop=10; %Número de população para otimizar os parâmetros do AE

%Número de testes para classificação
N=10;

%% 2) Carregamento de dados
%Descomentar aquele a ser utilizado
% 2.1 - Sinais no domínio do tempo
[TrainData,ValidationData,TestData,OptimizationData]=CarregaSinais_Z24_Tempo(num_col,num_points,num_ac,n_tr,caminho);

% 2.2 - Sinais no domínio da frequência
% [TrainData,ValidationData,TestData,OptimizationData]=CarregaSinais_Z24_Frequencia(num_col,num_points,num_ac,n_tr,caminho,T,hz);

% 2.3 - Sinais no domínio do tempo somados com o da frequência
% [TrainData,ValidationData,TestData,OptimizationData]=CarregaSinais_Z24_FrequenciaETempo(hz,T,num_col,num_points,num_ac,n_tr,caminho);

clear n_val num_col num_points 

%Caso não queira rodar a otimização, inserir os parâmetros da arquitetura
%da rede nesse local.
% hpObj=HyperparametersAED();
% hpObj.setHyperparametersAED('AutoencoderType',TipoAE,'LayersEncoder', ...
%     CamadaCod,'NeuronsEncoder',1055, ...
%     'LayersDecoder',CamadaDec,'NeuronsDecoder',1042,...
%     'WeightingKL',5, ...
%     'LearningRate',9390*10^(-7), ...
%     'MiniBatchSize',28, ...
%     'NumberEpoch',95,...
%     'LatentDim',N_latent)

%% 2) Otimização
[hpObj]=Otimizacao(ger,pop,TipoAE,CamadaCod,CamadaDec,N_latent,NeuCod,NeuDec,NumEpoc,OptimizationData);

clear OptimizationData ger pop NeuCod TipoAE CamadaCod CamadaDec N_latent NeuDec NumEpoc

%% 3) Classificação dos dados
%Descomentar aquela métrica a ser utilizada
% 3.1 - MSE com a Distância de Mahalanobis
[acertosD0,errosD0,errosDanos,acertosDanos] = Classificacao_Z24_MSE_Mahalanobis(N,hpObj,TrainData,ValidationData,TestData,n_test,n_tr,num_ac);

% 3.2 - MSE
% [acertosD0,errosD0,errosDanos,acertosDanos] = Classificacao_Z24_MSE(N,hpObj,TrainData,ValidationData,TestData,n_test,n_tr,num_ac);

