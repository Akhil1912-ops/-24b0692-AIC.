PROBLEM STATEMENT-
-""YOU SHOULD TRAIN A MODEL WHICH EXTRACTS THE DETAILS FROM THE GIVEN PHOTO"".
and classify and verify them

models-Tesseract OCR or TrOCR

QUANTITATIVE FORMULATION:-
The submitted model should be evaluvated by-
1.accuracy
2.model should financally fesible to check large number of ids
3.any kind of creative work they have done which can help this project in further stages

DATA SOURCING:-
(we are not available with having large number of data on this id cards.so lets use similar datasets)
datasets are ---
FUNSD(From Understanding in Noisy Scanned Documents)
CORD(1000+ images of receips)

PRE-PROCESSING TECHNIQUES-

convert image to grayscale 
resizing and normalization 


MODEL SELECTION -

tesseract ocr or TrOCR(both are open source and pre trained on the requirements)


TRAINING STRATEGIES-

as the models are already good to extract the text.fine-tune them on something more simlar to iD'S


COMPUTE REQUIREMENTS-

we can train these models on google collab using GPU and TPU.
so computational requirements are google collab.


team roles-(lets assume a team of 3) and milestones
week 1-
 member 1- collect the data and convert them and normalize them as needed for the model you choose
 member 2 and 3 - - decide what model to use and load the model
week 2 to week 4-
train the model on the data and increase the accuracy


 deliverables-
 submit the model and the whole python file with code
 and specify if any kind of specifications which improved the performance

 risk mitigation-
 the data sets like FUND AND CORD are not exactly id's .so the models may overfit due to this data.using
 data which can be similar to id's can help













