# R-CNN for template matching

#### Noli Manzoni, Micheal Denzler

If you want to have more information about our implementaiton please look at project description in the pdf file.

## How to use

### Get training data

To get the training data please execute the `mask_creator.py` script but first create an empty folder called `dataset` and put the templates under `images/templates`and backgrounds under ìmages/templates`.

### Train model

To train the model execute the file `model.py`.

### Test model
To test the model execute the `eval_model.py` file. This script will take the first element inside `dataset/images` and it will output the annotated image in the main directory.

### Contacts 

If you have any doubts please contact us at noli.manzoni@usi.ch or michael.denzler@usi.ch

## Results
Each template is annotated with a different color.

![results](https://github.com/raikilon/r-cnn-template-matching/blob/master/images/results/trainig_on_rooms/8.png)

