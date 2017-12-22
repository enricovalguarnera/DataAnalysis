close all
clear all

%Carico il dataset
X=caricaDataset('parkinsons.txt');

%Memorizzo la colonna contenente lo status (0/1)
E = num2str(X(:,17)); 
E = num2cell(E);

%Elimino la colonna 17
X(:,17)=[];

%Valori di m e perc, che variano per fare assumere valori diversi alla
%classificazione..
m = 18:-1:12; 
perc = 0.5:0.01:0.8;  

leg=cell(1,length(m));

for i=1:length(m)
    
    leg{i} = strcat('m=', num2str(m(i)));
    %Applico la funzione PCA al dataset X, trovando il dataset proiettato CX
    %moltiplicando la matrice di passaggio con il dataset originario X (passo gia
    %compreso nell'algoritmo PCA
    CX= PCA(X,m(i));
    for j=1:length(perc)
        %A questo punto non rimane che classificare....
        rCx(j) = bayesian_classifier(CX,E,perc(j));
    
    end 
    %plot dove sulle odinate vi è il riconoscimento sulle ascisse i valori
    %della percentuale,  tutto al variare delle m componenti
    %principali calcolate tramite l'algoritmo PCA
    title('Riconoscimento dal dataset CX al variare di m');
    plot(perc,rCx)
    hold on
    xlabel('Cardinalita Training Set - (Perc)');
    ylabel('Riconoscimento');
end
legend(leg);

figure();
for i=1:length(m)
     leg{i} = strcat('m=', num2str(m(i)));
    %Applico la funzione rprojection per ottenere il dataset proiettato PX
    PX = rprojection(X,m(i));
    for j=1:length(perc)
        %A questo punto non rimane che classificare....
        rCx(j) = bayesian_classifier(PX,E,perc(j));
    end 
    
    %plot dove sulle odinate vi è il riconoscimento sulle ascisse i valori
    %della percentuale,  tutto al variare delle m componenti
    %principali calcolate tramite l'algoritmo rprojection
    title('Riconoscimento dal dataset PX al variare di m');
    plot(perc,rCx)
    hold on
    xlabel('Cardinalita Training Set - (Perc)');
    ylabel('Riconoscimento');

end
legend(leg);