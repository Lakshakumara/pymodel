"""SLP - machine-learning-production
Usage:
    sp version
    sp train <grade>
    sp predict <data>
    sp exam
    sp fexam
    sp tma
    sp tmasum
    sp ass
    sp assavg
    sp grade <status> [-s]
    sp drop
    sp complete
    sp fail
    sp passrate
    sp failrate
    sp clkps
    sp merge
    sp heatmap
    sp model
    
    sp train <dataset-dir> <model-file> [--vocab-size=<vocab-size>]
    sp ask <model-file> <question>
    sp (-h | --help)
Arguments:
    <data>          1D numpy array eg: [90, 1, 2.5] 
    <grade>         Distinction, Pass, Fail, Withdrawn 
    <dataset-dir>   Directory with dataset.
    <model-file>    Serialized model file.
    <question>      Text to be classified.
Options:
    -s                         slice 
    --vocab-size=<vocab-size>  Vocabulary size. [default: 10000]
    -h --help                  Show this screen.
Commands:
  exam       
  fexam      
"""
from docopt import docopt

import os
from sklearn.metrics import classification_report

from slp import DumbModel, Dataset, OULAD_data, Model

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
def pp(a, args):
    print(a)
    n=[float(x) for x in args.split(',')]   
    print(n)
     
def main():
    arguments = docopt(__doc__)
    ou = OULAD_data()
    m= Model()
    if arguments['version']:
        return "1.0.0"
    if arguments['train']:
        return m.train(arguments['<grade>'], arguments['<grade1>'])
    
    if arguments['predict']:
        #print("in prediction block")
        n=[float(x) for x in arguments['<data>'].split(',')] 
        return m.predictionRF("rf_w_d", n)
        
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
        return ou._grade(arguments['<grade>'], arguments['-s'])
    
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
    
    elif arguments['model']:
        return m.train()

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