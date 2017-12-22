function [ X ] = caricaDataset ( nome )
% Tramite questa funzione carico il Dataset ”Parkinsons”,  che rappresenta 
% i valori numerici di 22 attributi estratti da 197 registrazioni vocali di
% 2 gruppi di persone, il primo affetto dal morbo di parkinson, l’altro no.
% Il dataset si presenta come una matrice V di 197 righe e 23 colonne.
% La funzione prende in input il nome del file e restituisce in output una
% matrice X 197x23.

% Carico il Dataset.
X= dataset('file',nome,'Delimiter',',');
% %Salvo la prima riga e la prima colonna
Y=X(:,1);
Z=X(1,:);

%elimino la prima colonna
X(:,1)=[];

%converto la matrice in double
X=double(X);