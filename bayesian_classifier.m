function r = bayesian_classifier(X,E,perc)
%bayesian_classifier Classificatore bayesiano
%   Dato in input il dataset X ed il vettore di celle E con le etichette delle
%   classi di appartenenza per ogni elemento di X, bayesian_classifier
%   produce, a partire da un Training Set estratto opportunamente dal 
%   dataset sulla base del valore di perc, una etichettatura del Testing
%   Set (formato dagli elementi restanti del dataset) e ne restituisce in
%   output il riconoscimento r  

%N è il numero degli elementi (righe) del dataset
[N ~]=size(X);

%C è il vettore di celle che contiene le classi distinte in cui E
%è suddiviso, in questo caso 0/1
C=unique(E);

%K è il numero delle classi
K=length(C);

%p è un vettore adibito a contenere le probabilità a priori p(C_i)
p=zeros(1,K);

%n è un vettore che memorizza nel suo i-esimo elemento la cardinalità del
%Training Set corrispondente alla classe i-esima
n=zeros(1,K);

%TrS è il Training Set: si tratta di un vettore di celle dove TrS{i} è 
%costituito dagli elementi prelevati dalla classe C{i} 
TrS=cell(1,K);

%TeS è il Testing Set: si tratta di un vettore di celle nel quale TeS{i} 
%è costituito dagli elementi rimanenti dalla classe C{i}
TeS=cell(1,K);

%mu è un vettore cella che memorizza le medie calcolate sui Training Set
%TrS{i}
mu=cell(1,K);

%sig è un vettore cella che memorizza le covarianze calcolate sui Training 
%Set TrS{i}
sig=cell(1,K);

%per ciascuna classe...
for k=1:K
    %identificazione degli indici degli elementi di classe C{k} 
    I=find(strcmp(E, C{k}) == 1);
    
    %calcolo della cardinalità del Training Set TrS{k} ottenuta come
    %percentuale fissata (perc) della cardinalità dell'intera classe C{k}
    n(k)=round(perc*length(I));
    
    %calcolo della probabilità a priori P(C_k), secondo interpretazione
    %frequentista (ossia numero elementi classe C{k} su dimensione dataset)
    p(k)=length(I)/N;
    
    %esegue una permutazione degli indici 
    permI = randperm(length(I));
    %calcola un nuovo vettore permI che ha numro di elementi minore , ma
    %contiene n(k) indici che esistono in I
    permI = permI(1:n(k));
    
    %composizione del Training Set per la classe C{k} attraverso la
    %selezione dei suoi primi n(k) elementi 
    %TrS{k}=X(I(1:n(k)),:);  originale
    TrS{k}= X(I(permI),:);
    
    %stima dei parametri mu e sigma dal Training Set TrS{k}
    mu{k}=mean(TrS{k},1);
    sig{k}=cov(TrS{k});
    
    %composizione del Testing Set per la classe C{k} attraverso la
    %selezione degli elementi da n(k)+1 in poi
    TeS{k}=X(I(n(k)+1:end),:);    
end

%R è il vettore cella che contiene le etichette delle classi cui gli
%elementi del Testing Set sono assegnati dal classificatore bayesiano
R=cell(1,K);

%per ogni classe...
for k=1:K
    %per ogni elemento del Testing Set di classe C{k}
    TeSlength = size(TeS{k},1);
    for i=1:TeSlength
        %prelevamento dell'i-esimo elemento di TeS{k}
        x=TeS{k}(i,:);
        
        %v è un vettore che contiene le probabilità a posteriori P(C_k|x)
        %dalle quali sarà possibile, tramite applicazione della regola del
        %maximum a posteriori probability (map), discriminare la classe
        v=zeros(1,K);
        
        %per ogni classe...
        for j=1:K
            %calcolo delle probabilità a posteriori attraverso la formula
            %di bayes (o meglio del suo numeratore); sono assunte come
            %likelihood P(x|C_k) delle normali multivariate di parametri mu
            %e sigma quelli stimati dai Training Set TrS{j} per ogni j
            v(j)=p(j)*mvnpdf(x,mu{j},sig{j});
            
        end
        
        %determinazione del massimo in v
        [m c]=max(v);
        
        %è scelta come etichetta della classe da assegnare all'i-esimo 
        %elemento del Testing Set TeS{k} quella cui corrisponde la massima 
        %probabilità a posteriori 
        R{k}(i)=c;
    end
end

%calcolo del riconoscimento per l'etchettatura prodotta, in particolare
%riconosce e trova tutti gli elementi del dataset che hanno etichetta
%appartenente alla classe i-esima
r=zeros(1,K);
for i=1:K
 r(i)=length(find(R{i}==i))/(size(TeS{i},1));
end
r=sum(r)/K;

end

