"""SLP - machine-learning-production
Usage:
    slp-cli exam
    slp-cli fexam
    slp-cli tma
    slp-cli tmasum
    slp-cli ass
    slp-cli assavg
    slp-cli grade <status> [-s]
    slp-cli drop
    slp-cli complete
    slp-cli fail
    slp-cli passrate
    slp-cli failrate
    slp-cli clkps
    slp-cli merge
    slp-cli heatmap
    
    slp-cli train <dataset-dir> <model-file> [--vocab-size=<vocab-size>]
    slp-cli ask <model-file> <question>
    slp-cli (-h | --help)
Arguments:
    <status>       Distinction, Pass, Fail, Withdrawn 
    <dataset-dir>  Directory with dataset.
    <model-file>   Serialized model file.
    <question>     Text to be classified.
Options:
    -s                         slice 
    --vocab-size=<vocab-size>  Vocabulary size. [default: 10000]
    -h --help                  Show this screen.
"""
from docopt import docopt

import os
from sklearn.metrics import classification_report

from slp import DumbModel, Dataset, OULAD_data

def train_model(dataset_dir, model_file, vocab_size):
    print(f'Training model from directory {dataset_dir}')
    print(f'Vocabulary size: {vocab_size}')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dset = Dataset(train_dir, test_dir)
    X, y = dset.get_train_set()

    model = DumbModel(vocab_size=vocab_size)
    model.train(X, y)

    print(f'Storing model to {model_file}')
    model.serialize(model_file)

    X_test, y_test = dset.get_test_set()
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

def ask_model(model_file, question):
    print(f'Asking model {model_file} about "{question}"')

    model = DumbModel.deserialize(model_file)

    y_pred = model.predict_proba([question])
    print(y_pred[0])

def exam():
    print("Exam data printing ")
    
def main():
    arguments = docopt(__doc__)
    ou = OULAD_data()
    if arguments['exam']:
        print("Exam activities")
        return ou._exam()
        
    elif arguments['tma']:
        print("All other activities")
        return ou._exam(False)
    
    elif arguments['tmasum']:
        print("Summary of other activities")
        return ou._no_of_tma()
    
    elif arguments['ass']:
        print("Student Assignments Data")
        return ou._stud_ass()
    
    elif arguments['assavg']:
        print("Final assessment average per student per module")
        return ou._assement_avg_ps_pm()
    
    elif arguments['grade']:
        print("student final grade")
        return ou._grade(arguments['<status>'], arguments['-s'])
    
    elif arguments['complete']:
        print("student has passesed final exam")
        return ou._complete()
    
    elif arguments['passrate']:
        print("Pass rate per student per module")
        return ou._pass_rate_ps_pm()
    
    elif arguments['failrate']:
        print("Fail rate per student per module")
        return ou._fail_rate_ps_pm()
    
    elif arguments['clkps']:
        print("General average per student per module")
        return ou._avg_clk_student()
    
    elif arguments['merge']:
        print("Merge all together")
        return ou._merge()
    
    elif arguments['heatmap']:
        return ou._heatmap()
    
    elif arguments['fexam']:
        return ou._final_exam_score()
    
    elif arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>'],
                    int(arguments['--vocab-size'])
        )
    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<question>'])

if __name__ == '__main__':
    main()