1.Purpose
    For this project, the work includes three parts.
    The first part is to do data-cleaning and pre-processing of rare dataset.
    The second part is to build a model for university recommender system.
    The thrid part is to give an example to call the build-in api from another file.

2.Call the prection method
    To get a predicition result, the third part given a example.
    First import required library.
    input:
        list of four scores:
            1.SAT Critical Reading score, 
            2.SAT Math score,
            3.SAT Writing score, 
            4.ACT Composite score
    output:
        two lists, which respectively contains:
            nine predicted schools' names and 
            nine predicted schools' scores in dataset. 
            
3Descirbtion of all files:
    1.college_prediciton_processing.ipynb(not need to run)
        This file is for the first part. The results is three datasets:
            data_df_25.csv
            data_df_75.csv
            data_df_mean.csv
            
    2.model_and_evaluation.ipynb(not need to run)
            This file is for the second part. The results is three ideal thresholds       for three datasets:
            

    3.predicition.ipynb
        The thrid part is to give an example to call the build-in api from another .ipynb file.
        
    4.IPEDS_data.xlsx
        original dataset.
    
    5.data_df_25.csv, data_df_75.csv, data_df_mean.csv:
        cleaned datasets. These datasets
        
    6.threshold.txt
        ideal threshold for model
        
    7.README.txt
