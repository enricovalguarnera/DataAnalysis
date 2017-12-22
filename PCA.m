function CX = PCA( X , m)

%il numero di componenti che spiegano la percentuale della varianza è dato
%in input (m)

%calcolo della matrice di covarianza del dataset
sig=cov(X);

%calcolo degli autovalori e autovettori della matrice di covarianza
%sig.V matrice che ha per colonne gli autovettori, D matrice diagonale
%che ha per elementi sulla diagonale gli autovalori corrispondenti
%agli autovettori individuati in V.
[V,D]=eig(sig);

%disposizione degli autovalori che si trovano sulla diagonale della
%matrice D in un vettore d
d=diag(D);

%ordiniamogli autovalori in ordine decrescente, s è il vettore che contiene
%gli autovalori ordinati, i contiene la permutazione degli indici di d che
%realizza l'ordinamento
[s,i]=sort(d,'descend');

%troviamo la matrice di passaggio M dato che conosciamo gia m, poichè 
%dato in input. La matrice viene costruita selezionando solo m autovettori
%corrispondenti agli m autovalori. Dal calcolo si ottiene una matrice con s
%righe e m colonne.
M=V(:,i(1:m));

%proiettiamo i dati nel nuovo spazio , otteniamo il dataset proiettato CX
CX=X*M;

end

