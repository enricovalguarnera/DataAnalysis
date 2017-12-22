function PX = rprojection( X , m )

%Questa è la funzione che implementa il metodo delle proiezioni random.
%Lo scopo è proiettare il dataset da uno spazio dei dati a d dimensioni in
%uno spazio dei dati a d' dimensioni dove d' < d. 
%Per fare ciò bisogna moltiplicare il dataset per la matrice di passaggio P

%Prendo le dimensioni del dataset X , righe e colonne
[r,c]=size(X);

%Creo una matrice temporanea Rt di dimensione c * m con valori compresi fra
%0 e 1
Rt=rand(c,m);

%Inizializzo il valore della matrice A
A = zeros(c,m);

%La matrice A è la matrice di Achlioptas e viene calcolata alla base delle
%probabilità designate nel testo della tesina. 
%Vado ad inserire il valore -sqrt(3) se il valore è minore di 1/6 e vado a
%inserire il valore sqrt(3) se il valore è maggiore di 5/6
A(find(Rt<1/6))=-sqrt(3);
A(find(Rt>5/6))=sqrt(3);
%Inizializzo il valore della matrice in output C
PX = zeros(c);
%La matrice di passaggio P viene calcolata tramite la seguente formula
P = A/sqrt(m);

%La matrice PX, che corrisponde alla proiezione del dataset 
PX = X*P;
end
