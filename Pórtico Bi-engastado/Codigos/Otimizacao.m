function [hpObj] = Otimizacao(ger,pop,TipoAE,CamadaCod,CamadaDec,N_latent,NeuCod,NeuDec,NumEpoc,OptimizationData)
%% 3) OTIMIZAÇÃO DOS HIPERPARÂMETROS
optiSettings=HPOsettingsAED() %Criação de um AE padrão para ser otimizado
optiSettings.settingsGA %Mostra as configurações padrões da arquitetura da rede 

%O tempo computacional aumenta de acordo com o número de indivíduos por
%geração e o número máximo de gerações.

optiSettings.setValuesGA(ger,pop) %Criação de um AE modificado Máximo de 3 gerações
% com uma população de 10
optiSettings.settingsGA %Mostra as configurações modificadas da arquitetura da rede 

optiSettings.setSettingsAED('AutoencoderType',TipoAE,'LayersEncoder',CamadaCod,'LayersDecoder',CamadaDec,'LatentDimension',N_latent)

optiSettings.optimizableVariables %Mostra as configurações otimizadas das variáveis

optiSettings.setRangesOptimization('NeuronsEncoderLayer1',NeuCod,'NeuronsDecoderLayer1',NeuDec,'NumberEpochs',NumEpoc)


optiSettings.optimizableVariables %Exibição das variáveis alteradas

%Otimização usando os dados de OptimizationData e as configurações de
%otimização definidas
[population,bestFitness,avgFitness,bestIndividual,aed,fitness] = GA_HPO_AED(OptimizationData, optiSettings)

%% 4)CRIAÇÃO DO AUTOENCODER COM HIPERPARÂMETROS OTIMIZADOS
%Inicialização de um AE pré-configurado
%Criação de um autoencoder personalizado com os parâmetros otimizados
bestIndividualLast=bestIndividual{length(bestIndividual)}{1}

hpObj=HyperparametersAED();
hpObj.setHyperparametersAED('AutoencoderType',TipoAE,'LayersEncoder',CamadaCod,'NeuronsEncoder',bestIndividualLast{5},'LayersDecoder',CamadaDec,'NeuronsDecoder',bestIndividualLast{4},...
    'WeightingKL',bestIndividualLast{1}, ...
    'LearningRate',bestIndividualLast{2}*10^-7, ...
    'MiniBatchSize',bestIndividualLast{3}, ...
    'NumberEpoch',bestIndividualLast{length(bestIndividualLast)},...
    'LatentDim',N_latent)

hpObj.Hyperparameters %visualização dos parâmetros



