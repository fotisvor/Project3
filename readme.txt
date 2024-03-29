Εργασία 3 Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα 
Φώτιος Βορλόου 1115201900026
Βάιος Καραμήτρος 1115201900073

Ερώτημα Α:
Ο παρεχόμενος κώδικας υλοποιεί έναν convolutional autoencoder για τη μείωση της διάστασης των δεδομένων εισόδου.

Επεξήγηση κώδικα:
Βιβλιοθήκη: Εισαγωγές:

idx2numpy: Χρησιμοποιείται για τη μετατροπή δεδομένων από τη μορφή αρχείου IDX.
numpy: Θεμελιώδες πακέτο για επιστημονικούς υπολογισμούς με την Python.
matplotlib.pyplot: Βιβλιοθήκη για τη δημιουργία οπτικοποιήσεων.
keras.layers: Εξαρτήματα για τη δημιουργία μοντέλων βαθιάς μάθησης με τη χρήση του Keras.
keras.models: Λειτουργικό API για τον ορισμό σύνθετων μοντέλων στο Keras.
sklearn.model_selection.train_test_split: Χρησιμοποιείται για το διαχωρισμό του συνόλου δεδομένων σε σύνολα train και validation.
Φόρτωση και προεπεξεργασία δεδομένων:

idx2numpy.convert_from_file: Μετατρέπει τα δεδομένα από αρχείο IDX σε πίνακα NumPy.
Αναδιαμορφώνει τα δεδομένα ώστε να ταιριάζουν με το αναμενόμενο σχήμα εισόδου του autoencoder (εικόνες 28x28).
Κανονικοποιεί τις τιμές των pixels στο εύρος [0, 1].
Χωρίζει το σύνολο δεδομένων σε σύνολα train και validation (90% train, 10% validation).

Ορισμός μοντέλου autoencoder:

Η αρχιτεκτονική αποτελείται από στρώματα convolutional και pooling για την κωδικοποίηση και στρώματα upsampling για την αποκωδικοποίηση.
Ο autoencoder μειώνει την είσοδο σε μια αναπαράσταση 128 διαστάσεων.
Ο autoencoder ανακατασκευάζει την είσοδο από τη μειωμένη αναπαράσταση.
Το μοντέλο χρησιμοποιεί τον βελτιστοποιητή Adam και binary cross-entropy loss.

Εκπαίδευση του autoencoder:
Το μοντέλο εκπαιδεύεται για 10 epochs με μέγεθος batch 128.
Οι απώλειες train και validation παρακολουθούνται για την αξιολόγηση της απόδοσης του μοντέλου.

Αποφυγή overfitting:

To overfitting αντιμετωπίζεται με την παρακολούθηση της απώλειας validation κατά τη διάρκεια του train.
Η αρχιτεκτονική του μοντέλου περιλαμβάνει στρώματα μέγιστης συγκέντρωσης, τα οποία βοηθούν στην εκμάθηση ισχυρών χαρακτηριστικών και στη μείωση του overfitting.
Η χρήση στρωμάτων εγκατάλειψης ή τεχνικών κανονικοποίησης εξεταστηκε για την περαιτέρω πρόληψη της overfitting, αλλά δεν βρήκαμε σημαντικό αποτέλεσμα.

Κωδικοποίηση δεδομένων εισόδου και ερωτήματος:

Ο autoencoder χρησιμοποιείται για την κωδικοποίηση τόσο των δεδομένων εισόδου όσο και των δεδομένων queries.
Οι κωδικοποιημένες αναπαραστάσεις αποθηκεύονται στα αρχεία reducedinput.dat και reducedquery.dat.

Απεικόνιση του ιστορικού train:

Για οπτικοποίηση δημιουργείται μια γραφική παράσταση των loss train και validation κατά τη διάρκεια των epochs.
Έξοδος train:
Η έξοδος train δείχνει ότι το μοντέλο μαθαίνει, με φθίνουσα απώλεια με την πάροδο των εποχών. Η απώλεια validation μειώνεται επίσης, γεγονός που υποδηλώνει ότι το μοντέλο δεν προσαρμόζεται υπερβολικά στα δεδομένα train.
Περαιτέρω σκέψεις:
Ο αριθμός των εποχών, το μέγεθος της παρτίδας και οι παράμετροι της αρχιτεκτονικής του μοντέλου μπορούν να ρυθμιστούν για καλύτερη απόδοση.
Μπορούν να διερευνηθούν πρόσθετες τεχνικές, όπως η αύξηση των δεδομένων ή η προσαρμογή του ρυθμού μάθησης για την ενίσχυση της γενίκευσης.
Πειραματιστείτε με διαφορετικές συναρτήσεις ενεργοποίησης, συναρτήσεις απωλειών και βελτιστοποιητές για πιθανές βελτιώσεις.

Ερώτημα Β:
Τα περισσότερα tests έγιναν για αριθμό MNIST_Images 60.000 Και train set 10.000 Καθώς πιστεύαμε οτι μια καλύτερα εικόνα των διαφορών με τις μειωμένες διαστάσεις . Αρχικά έγιναν προσαρμογές στο πως κάνουμε Load το Mnist dataset διότι πλέον οι χώροι απο εκέι που ήταν 28*28 (784 διαστάσεις) πλέον μετά τη χρήση του reduce.py έχουμε εικόνες 4x4x8 (128) διαστάσεων. Και στους 2 κώδικες παρατηρήθηκε σημαντική αύξηση στην ακρίβεια (πχ αρχικά το ΑF Κυμαινόταν κοντά στο 1.0 και στο reduced dimensions αυξανόταν πολύ πάνω απο αυτό) και στη σημαντική μείωση στον χρόνο (πχ στο mrng για 5 γείτονες και 10 υποψήφιους στα 1κ Images απο 1 σχεδόν λεπτό το πρόγραμμα έκανε περίπου 10 δευτερόλεπτα με τις μειωμένες διαστάσεις). 

Ερώτημα Γ:
Εδώ συναντήσαμε ένα σοβαρό πρόβλημα. Κατεβάζοντας από το eclass το παραδοτέο μας για την 1η εργασία παρατηρήσαμε ότι ο κώδικας του cluster δεν έτρεχε σωστά, παρόλο που όταν την στείλαμε και όταν εξεταστίκαμε για αυτή όλα πήγαιναν καλά. Έτσι μας είναι αδύνατο να τρέξουμε το clustering, όμως έχουμε κάνει τις απαραίτητες αλλαγές ώστε ο κώδικας να είναι συμβατός με την νέα διάσταση. Πιο συγκεκριμένα, προσθέθηκε η αντικειμενική συνάρτηση k-means, η οποία είναι η ελαχιστοποίηση του αθροίσματος των τετραγωνικών αποστάσεων από κάθε σημείο δεδομένων προς το κεντροειδές που του έχει ανατεθεί.Η συνάρτηση αυτή επαναλαμβάνει κάθε συστάδα και υπολογίζει το άθροισμα των τετραγωνικών αποστάσεων από κάθε σημείο δεδομένων σε αυτή τη συστάδα προς το κεντροειδές που του έχει ανατεθεί. Το τελικό αποτέλεσμα είναι η αντικειμενική συνάρτηση k-means. Τέλος, έχει τροποποιηθεί η sillouette, ώστε να υπολογίζει την ευκλείδια απόσταση στον αρχικό χώρο 28*28.
