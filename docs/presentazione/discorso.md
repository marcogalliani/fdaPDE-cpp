[titolo]
L'obiettivo di questo progetto è implementare all'interno di fdaPDE alcuni metodi di algebra lineare randomizzata per rendere più efficienti gli algoritmi di functional PCA presenti nella libreria. 

[slide fPCA regolarizzata]
In particolare, gli algoritmi in fdaPDE risolvono il problema di fPCA regolarizzata. Consideriamo un dataset funzionale, le cui righe sono funzioni definite su un dominio spaziale e valutate ad un insieme fissato di locazioni. Sia F la matrice delle M componenti principali e S la matrice degli scores relativa alle PC, il problema di fPCA può essere scritto come riportato in figura, dove il termine |X-SF^T| rappresenta la caratterizzazione della fPCA come migliore approssimazione di rango-M dei dati, e il secondo termini promuove la smoothness delle componenti estratte.

In fdaPDE esistono due algoritmi per calcolare la fPCA: sequenziale e monolitico. entrambi gli algoritmi coinvolgono il calcolo di una SVD truncata ad M, attualmente eseguito utilizzando l'algoritmo di Jacobi implementato in Eigen.

[slide intro-RandSVD]
Questo algoritmo è computazionalmente molto costoso. Inoltre calcola la decomposizione completa, anche se questa non è necessaria ai fini della fPCA. Per questo motivo lo scopo di questo progetto è implementare classe di algoritmi randomizzati per calcolare la SVD. Questi si basano su tre step: approsimazione del range della matrice originale, proiezione delle colonne della matrice originale sul range approssimato, ottenendo così un'approssimazione di rango ridotto, e manipolazione di questa approsimazione in una decomposizione standard, nel nostro caso la SVD.

[slide approx-range]
La randomizzazione entra in gioco nel primo step. Infatti, si può ottenere una approsimazione di rango di rango k del range della matrice input applicando la matrice A alla matrice omega, ottenuta campionando k vettori aleatori gaussiani. Si puo dimostrare che questo prodotto è maggiormente correlato scon i k left singular vectors dominanti di A. Tale correlazione è amplificata iterando il prodotto per AA^T, e riducendo  in questo modo l'impatto dei singular vector minori sulla qualita dell'approsimazione.  

[slide proiezione]
Ottenuta una buona approsimazione del range, rappresentata dalla matrice M, si proiettano le colonne di A sul range di M ortogonalizzando M ottenendo una approsimazione di rango ridotto di A. Infatti se definiamo la matrice residuale B come Q^TA notiamo che sia Q che B hanno rango k.

[slide SVD]
A partire dall'approsimazione di rango ridotto, si puo convertire questa decomposizione in una SVD calcolando la SVD di B e moltiplicando Q a U tilde senza introdurre ulteriori approsimazioni.

[slide RSI-RBKI]
Esistono due famiglie di algoritmi di SVD randomizzata, RSI e RBKI. I due algoritmi si differenziano in base allo spazio di proiezione che utilizzano. RSI utilizza l'ultimo prodotto calcolato, mentre RBKI utilizza tutto il sottospazione di Krylov contenetni i prodotti calcolati alle iterazioni precedenti.

[slide stopping criterion]
Per concludere, fino ad ora abbiamo considerato un numerato fissato di iterazioni. In practica, le iterazioni vengono terminate quando tutti i singular vectors estratti raggiungono l'accuratezza residua riportata in (1). Infatti, se (1) è soddisfatta un teorema garantisce l'esistenza di una perturbazione di A, A+E, con E controllata, tale la SVD-troncata calcolata attraverso l'algoritmo sia esattamente la SVD troncata di A+E. 

[slide intro-implementazione]
Passando all'implementazione, gli algoritmi di SVD randomizzata sono stati implementati all'interno del modulo di algebra lineare di fdapde. Inoltre, alcune modifiche sono state apportate al modulo di functional data analysis per integrare la SVD randomizzata e implementare l'estensione dei modelli di fPCA per missing data.

[slide RandSVD-Strategy pattern]
Dovendo implementare diversi algoritmi per che svolgono la stessa task, ossia la SVD randomizzata, ho utilizzato lo startegy pattern. In particolare, le classi implementate sono: una classe astratta RSVDStrategy, che rappresenta un astrazione per la task di SVD randomizzata, da cui ereditano le classi che implementano nel concreto gli algoritmi di SVD randomizzata. Oltre ai già citati RSI e RBKI sono presenti anche le loro versioni generalizzate. La classe RSVD ha un puntatore alla classe RSVDSTrategy che consente di selezionare a run-time l'algoritmo da utlizzare. 

[slide RSVD]
Inoltre questa è la classe che espone all'utente l'interfaccia per effetuare la SVD. Come l'implementazione di Eigen espone i getters per accedere ai singular values e sigular vectors. Inoltre espone due overloads del metodo compute attraverso cui la SVD è effettivamente calcolata. Tutto il lavoro è poi delegato all strategia puntata dalla classe. Infine la classe RSVD offre l'opportunità di definire un trait per riconoscere l'utilizzo di algoritmi di SVD randomizzata che verrà utilizzato per facilitare l'integrazion all'interno degli algoritmi di fPCA. 

[slide Implementazione attuale]
Attualmente gli algoritmi sono implementati in un solver RegularizedSVD definito all'estrno della classe FPCA. Questa classe è un template con un parametro SolutionPolycy ed implementa gli algoritmi monolitico e sequenziale specializzando la classe per i due casi. Per integrare la possibilità di utilizzare un algoritmo randomizzato per la SVD ho aggiunto un parametro template che specifica il solver da utilizzare per calcolare la SVD. Il dispatching della SVD è quindi effetuato utilizzano il compile time if insieme al trait precedemente definito.

[slide partially observed data]
Ora introduciamo brevemente l'estensione del modello di fPCA per dati parzialmente osservati. In particolare, data la matrice binaria W, che rappresenta la presenza o meno della funzione i alla locazione j, il problema di fPCA si puo riscrivere come prima, includendo il prodotto di Hadamard nel termine di data-fidelity. 

[slide algoritmo risoluzione]
Questo problema viene risolto utilizzando un procedimento iterativo in cui a ciascuna iterazione un problema di fPCA per dati completamente osservati viene risolto su un nuovo dataset, in cui dati mancanti vengono imputati con le soluzioni calcolate all'iterazione precedente. La presenza di molteplici problemi di fPCA richiede quindi molteplici SVD e rende ancora più vantaggioso l'utilizzo dell' SVD randomizzata in questo contesto. 

[slide implementazione]
Per implementare l'algoritmo un nuovo parametro che rappresenta la presenza di dati mancanti è aggiunto alla classe RegularizedSVD. Gli algoritmi vengono poi implementati come specializzazioni della classe.

[slide test randSVD]
Passando alla parte di test qui sono riportati i tempi di esecuzione di un test su matrici quadrate con singular values lentamente decresecenti, ossia il worst-case scenario per gli algoritmi randomizzati. Osserviamo come per matrici quadrate la SVD esatta implementata in eigen abbia complessità cubica, mentre RSI e RBKI hanno complessità quadratica.

[slide scikit-learn]
In questa slide è riportato un confronto tra un implementazione di RSI disponibile in scikit-learn e le mie implementazioni. Osserviamo che RSI batte consistentemente la sua analoga implementazione in python, mentre RBKI diventa inefficiente al crescere delle dimensioni della matrice.


[slide generazione dati fPCA]
Passiamo dunque ai test sull'integrazione all'interno di fPCA. I dati sono generati di test sono generati a partire dalle autofunzioni dell'operatore di diffusione anistropa. Una volta calcolate un numero di autofunzioni fissato (nel nostro caso 3), ogni funzione nel dataset è generata effetuando il sampling dello score relativo a ciascuna PC, dove ogni score è distribuito come una normale con varianza decresecente nel indice della PC. In ultimo, del rumore gaussiano viene aggiunto per complicare la situazione.

[slide fPCA-computational bottlenecks]
Riepiloghiamo brevemente i bottleneck computazionali negli algoritmi di fPCA

[slide statistical units]
AL crescere di unità statistiche la SVD diventa un bottleneck per entrambi gli algoritmi, l'utilizzo di una SVD randomizzata permette di ridurre fortemente l'impatto di questo bottleneck. Nel grafico è infatti riportata la percentuale di tempo impiegata nel clacolare la SVD sul runtime complessivo

[slide mesh nodes]
Al crescere dei nodi della mesh si ossrva che la SVD smette di essere il bottleneck principale, specialmente nella monolitica. COmunque l'utillizo di una SVD randomizzata riduce la percentual di impiegato per calcolare la SVD

[slide missing]
In ultimo, riportiamo alcuni test nel caso di dati parzialmente osservati. In particlare il caso di censoramento dipendente dei dati è stato considerato nella fase di test, dove dati vicini tra loro nello spazio sono stati eliminati. 

In questo primo grafico riportiamo l'accuracy dei due algorimi e delle loro versioni con SVD randomizzata, confrontati con un metodo competitor (DINEOF). L'implementazione replica dunque i risultati ottenuti nel papaer in cui l'algoritmo di fPCA per dati parzialmente osservati è stato introdotto e le SVD randomizzate non introducono approssimazioni significative. Inoltre se osserviamo i tempi di esecuzioni osserviamo il guadagno in termine di efficienza, che porta la fPCA ad essere competitiva con il metodo competitor