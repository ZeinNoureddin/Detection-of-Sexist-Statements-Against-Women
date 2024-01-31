# Detection of Sexist Statements Against Women

A deep learning bidirectional GRU model with the attention mechanism for the automatic detection of sexist statements
commonly used at the workplace. The baseline model that we relied on is available [here](https://github.com/dylangrosz/Automatic_Detection_of_Sexist_Statements_Commonly_Used_at_the_Workplace).

## Running the project

We ran the project via Google Colab. To do that, upload the folder titled "Automatic_Detection_of_Sexist_Statements_Commonly_Used_at_the_Workplace-master" inside the "Code Files" folder to the drive. You will need to add the weights (available [here](https://drive.google.com/file/d/1vhnuVR1sXr4-3-d_CQAlB2AxVCbbwSGC/view?usp=sharing)), as well as the dataset file (use [shuffled_data.csv](https://drive.google.com/file/d/1n2iezfCwoWqMiRPhJaUgIdOix2Qh75b4/view?usp=sharing) for the regular dataset; [cleaned_general_hate_speech_dataset.csv](https://drive.google.com/file/d/18f9FN5Ftvf6NzTwbjoyj9IrJwh59vOnA/view?usp=sharing) is the dataset we used when fine-tuning and is attached for reference) to the empty folders titled "weights" and "data" in the "Code Files" folder, respectively. Also upload the "Running the Model.ipynb" file to the drive. Change the path of the folder in the second cell of the notebook to reflect where you uploaded the folder. Run the "Running the Model.ipynb" notebook to run the model. For context, "Running the Model.ipynb" runs "SexismDetectionMain3.py" to run the model.

## Dependencies

If you are running from Google Colab, you will only need to install the following dependencies:

- emoji==0.5.2
- contractions==0.0.18
- Faker==1.0.5
  Otherwise, install all the dependencies in the "requirements.txt" file (found in the Code Files folder).

## Notes

The file "SexismDetectionMain3.py" runs our best model. The file "Other Versions of the Model.py" (found in the Code Files zip) contains different model definitions that we used when we were experimenting with the model. \
If you want to change the path of the weights that are being loaded into the model, you can do that by navigating to line 408 in the "SexismDetectionMain3.py" file.
