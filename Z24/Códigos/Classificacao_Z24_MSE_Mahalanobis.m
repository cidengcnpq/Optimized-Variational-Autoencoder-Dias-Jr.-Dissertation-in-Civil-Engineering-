function [acertosD1,errosD1,errosdanos,acertosdanos] = Classificacao_Z24_MSE_Mahalanobis(N,hpObj,TrainData,ValidationData,TestData,n_test,n_tr,num_ac)
acertosD1=0;errosD1=0;errosdanos=0;acertosdanos=0;

for i=1:N
    %% 5)TREINAMENTO DO AUTOENCODER COM HIPERPARÂMETROS
    aeCustomHP=trainAutoencoderDeep(TrainData,hpObj)
    
    %% 6)RECONSTRUÇÃO DOS DADOS
    
    %Reconstruindo os dados de treinamento
    [train_reconstructedOutput,train_latentRepresentation,train_reconstructionErrorPerSampleNormalized,train_reconstructionErrorPerChannelNormalized,train_failedIndex,train_originalInput]=predictAutoencoderDeep(TrainData, aeCustomHP);
    
    %Reconstruindo os dados de validação
    [val_reconstructedOutput,val_latentRepresentation,val_reconstructionErrorPerSampleNormalized,val_reconstructionErrorPerChannelNormalized,val_failedIndex,val_originalInput]=predictAutoencoderDeep(ValidationData, aeCustomHP);
    
    %Construindo os os dados de teste
    [test_reconstructedOutput,test_latentRepresentation,test_reconstructionErrorPerSampleNormalized,test_reconstructionErrorPerChannelNormalized,test_failedIndex,test_originalInput]=predictAutoencoderDeep(TestData, aeCustomHP);
   
    
    %% 7) CÁLCULO DOS RESÍDUOS ATRAVÉS DO MSE
    
    M_Input=[train_originalInput;val_originalInput;test_originalInput]; %Constrói uma matriz com todos os Input's 
    M_Output=[train_reconstructedOutput;val_reconstructedOutput;test_reconstructedOutput]; %Constrói uma matriz com todos os Output's 
    
    for k=1:length(M_Input);
        MSE(k,1)=mean((M_Input{k}-M_Output{k}).^2); %Preenche uma matriz coluna com todos os erros (1 ponto por cada ensaio)
    end
    
 %% 8) CLASSIFICAÇÃO POR MAHALANOBIS
     defInd = [1:n_tr];
    x = MSE;                   % Matriz a ser comparada
    Xd = MSE(defInd,1);        % Matriz composta pelos dados de treino e validação
    mu = mean(Xd,1);          % Média da Matriz composta pelos dados de treino e validação
    S = cov(Xd);              % Covariância da Matriz composta pelos dados de treino e validação
    invS = pinv(S); % Inversa da Matriz composta pelos dados de treino e validação
  
            for j = 1:length(x)
                T(j) = sqrt((x(j,:)-mu) * invS * (x(j,:)-mu)'); % Mahalanobis distance
            end
    
    T=T';
    limiar=prctile(T(1:(n_tr),1),95,1); %Calcula um ponto que está acima de 95% dos resultados dos dados de treinamento
    
    %Contagem para a matriz de confusão
    dano1=T(T(1:n_test,1)<=limiar,1);
    a=length(dano1);
    b=n_test-length(dano1);

    danos=T(T(n_test+1:end,1)<=limiar,1);
    c=length(danos);
    d=3*n_test-length(danos);

    acertosD1=a+acertosD1;
    errosD1=b+errosD1;
    errosdanos=c+errosdanos;
    acertosdanos=d+acertosdanos;
   
    limiar=limiar*ones(1,length(M_Input)); %Cria uma matriz para plotagem de uma reta no gráfico mostrando o limiar
    
  figure()
    semilogy(1:n_tr,T(1:n_tr),'Color',[0.9290 0.6940 0.1250],'LineStyle','none','Marker','*')
    hold on
    semilogy((n_tr+1:n_test),T((n_tr+1):n_test),'Color',[0.6350 0.0780 0.1840],'LineStyle','none','Marker','*')
    hold on
    semilogy((n_test+1:n_test*2),T((n_test+1):2*n_test),'Color',[0 0.4470 0.7410],'LineStyle','none','Marker','*')
    hold on
    semilogy((2*n_test+1:n_test*3),T(2*n_test+1:n_test*3),'Color',[0.4940 0.1840 0.5560],'LineStyle','none','Marker','*')
    hold on
    semilogy((3*n_test+1:4*n_test),T(3*n_test+1:4*n_test),'Color',[0.4660 0.6740 0.1880],'LineStyle','none','Marker','*')
    hold on
    semilogy(limiar,'b','LineWidth',2)
    hold on
    xline([0 n_tr n_test n_test*2 n_test*3], '-', {'Treino - D1','Validação - D1', 'Teste - D2','Teste - D4','Teste - D5'})
    sgtitle(['Resíduos através do cálculo por MSE depois Mahalanobis - Acelerômetro ',num2str(num_ac)])
    % saveas(gcf,strcat(folder,filename),'jpg');

 clearvars -except acertosD1 errosD1 errosdanos acertosdanos N hpObj TrainData ValidationData TestData n_test n_tr num_ac i

end















    