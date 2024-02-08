function [acertosD0,errosD0,errosDanos,acertosDanos] = Classificacao_CincoDanos_MSE(N,hpObj,TrainData,ValidationData,TestData,n_test,n_tr,num_ac)
acertosD0=0;errosD0=0;errosDanos=0;acertosDanos=0;

for i=1:N
    %% 1)TREINAMENTO DO AUTOENCODER COM HIPERPARÂMETROS
    aeCustomHP=trainAutoencoderDeep(TrainData,hpObj)
    
    %% 2)RECONSTRUÇÃO DOS DADOS
    
    %Reconstruindo os dados de treinamento
    [train_reconstructedOutput,train_latentRepresentation,train_reconstructionErrorPerSampleNormalized,train_reconstructionErrorPerChannelNormalized,train_failedIndex,train_originalInput]=predictAutoencoderDeep(TrainData, aeCustomHP);
    
    %Reconstruindo os dados de validação
    [val_reconstructedOutput,val_latentRepresentation,val_reconstructionErrorPerSampleNormalized,val_reconstructionErrorPerChannelNormalized,val_failedIndex,val_originalInput]=predictAutoencoderDeep(ValidationData, aeCustomHP);
    
    %Construindo os os dados de teste
    [test_reconstructedOutput,test_latentRepresentation,test_reconstructionErrorPerSampleNormalized,test_reconstructionErrorPerChannelNormalized,test_failedIndex,test_originalInput]=predictAutoencoderDeep(TestData, aeCustomHP);
    
    %% 3) VISUALIZAÇÃO DOS SINAIS - COMPARAÇÃO ENTRE OS ORIGINAIS E RECONSTRUÍDOS
    % Comparação entre dados de treino
    % filename3=[filename,'3_11'];
    % titulo3='Originais e reconstruídos - treinamento';
    % plotalldata_comp_trans1D_2(train_originalInput,train_reconstructedOutput,n_tr,folder,filename3,titulo3)
    % 
    % % 
    % filename4=[filename,'4_11'];
    % titulo4='Sinal reconstruído - treinamento';
    % plotalldata_trans1D_2(train_reconstructedOutput,n_tr,folder,filename4,titulo4)
    % % 
    % % Comparação entre dados de validação
    % filename5=[filename,'5_11'];
    % titulo5='Originais e reconstruídos - validação';
    % plotalldata_comp_trans1D_2(val_originalInput,val_reconstructedOutput,n_val,folder,filename5,titulo5)
    % % 
    % filename6=[filename,'6_11'];
    % titulo6='Sinal reconstruído - validação';
    % plotalldata_trans1D_2(val_reconstructedOutput,n_val,folder,filename6,titulo6)
    % 
    % % Comparação entre dados de teste
    % filename7=[filename,'7_11'];
    % titulo7='Originais e reconstruídos - teste';
    % plotalldata_comp_trans1D_2(test_originalInput,test_reconstructedOutput,n_test*2,folder,filename7,titulo7)
    % % 
    % filename8=[filename,'8_11'];
    % titulo8='Sinal reconstruído - teste';
    % plotalldata_trans1D_2(test_reconstructedOutput,n_test,folder,filename8,titulo8)
    % %
    % filename9=[filename,'9_11'];
    % titulo9='Sinais originais e reconstruídos - D1';
    % plotalldata_comp_trans1D_2(test_originalInput(1:n_test),test_reconstructedOutput(1:n_test),n_test,folder,filename9,titulo9)
    % %
    % filename10=[filename,'10_11'];
    % titulo10='Sinais originais e reconstruídos - D2';
    % plotalldata_comp_trans1D_2(test_originalInput(n_test+1:end),test_reconstructedOutput(n_test+1:end),n_test,folder,filename10,titulo10)
    
    %% 4) CÁLCULO DOS RESÍDUOS ATRAVÉS DO MSE
    
    M_Input=[train_originalInput;val_originalInput;test_originalInput]; %Constrói uma matriz com todos os Input's 
    M_Output=[train_reconstructedOutput;val_reconstructedOutput;test_reconstructedOutput]; %Constrói uma matriz com todos os Output's 
   
    for k=1:length(M_Input);
        MSE(k,1)=mean((M_Input{k}-M_Output{k}).^2); %Preenche uma matriz coluna com todos os erros (1 ponto por cada ensaio)
    end
    
    limiar=prctile(MSE(1:(n_tr),1),95,1); %Calcula um ponto que está acima de 95% dos resultados dos dados de treinamento
    
    %Contagem para a matriz de confusão
    dano0=MSE(MSE(1:n_test,1)<=limiar,1);
    a=length(dano0);
    b=n_test-length(dano0);
    
    danos=MSE(MSE(n_test+1:end,1)<=limiar,1);
    c=length(danos);
    d=4*n_test-length(danos);

    acertosD0=a+acertosD0;
    errosD0=b+errosD0;
    errosDanos=c+errosDanos;
    acertosDanos=d+acertosDanos;
   
    limiar=limiar*ones(1,length(M_Input)); %Cria uma matriz para plotagem de uma reta no gráfico mostrando o limiar
    
    
    %Visualização dos resultados
    figure()
    semilogy(1:n_tr,MSE(1:n_tr),'Color',[0.9290 0.6940 0.1250],'LineStyle','none','Marker','*')
    hold on
    semilogy((n_tr+1:n_test),MSE((n_tr+1):n_test),'Color',[0.6350 0.0780 0.1840],'LineStyle','none','Marker','*')
    hold on
    semilogy((n_test+1:n_test*2),MSE((n_test+1):2*n_test),'Color',[0 0.4470 0.7410],'LineStyle','none','Marker','*')
    hold on
    semilogy((2*n_test+1:n_test*3),MSE(2*n_test+1:n_test*3),'Color',[0.4940 0.1840 0.5560],'LineStyle','none','Marker','*')
    hold on
    semilogy((3*n_test+1:n_test*4),MSE(3*n_test+1:n_test*4),'Color',[0.4660 0.6740 0.1880],'LineStyle','none','Marker','*')
    hold on
    semilogy((4*n_test+1:n_test*5),MSE(4*n_test+1:end),'Color',[0.8500 0.3250 0.0980],'LineStyle','none','Marker','*')
    hold on
    semilogy(limiar,'b','LineWidth',2)
    hold on
    xline([0 n_tr n_test n_test*2 n_test*3 n_test*4], '-', {'Treino - D0','Validação - D0', 'Teste - D1','Teste - D2','Teste - D3','Teste - D4'})
    sgtitle(['Resíduos através do cálculo por MSE - Acelerômetro ',num2str(num_ac)])
    % saveas(gcf,strcat(caminho,filename),'jpg');

end


