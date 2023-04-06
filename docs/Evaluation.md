# Evaluate the Trained Checkpoints

* **NOTE**: The checkpoints and result files of VLPD are provided on [BaiduYun](https://pan.baidu.com/s/1rF8TEXybCdDUWO-HvzxbbQ?pwd=VLPD) and [GoogleDrive](https://drive.google.com/drive/folders/1rcGjK36zDZqxULoAztexupjxNlB0U4F6?usp=sharing). 
* If needed, place these files in path "output/city (or caltech)" and perform 2~3 and 3~5 for evaluations. 
* Evaluation of results on Caltech during training is also performed as [here](#evaluate-the-results-on-caltech) (start from step 3).

## Evaluate the results on CityPersons

1. Download the [checkpoint](#evaluate-the-trained-checkpoints) of our VLPD for CityPersons.

2. Run "sh test.sh" (please use the **upper** lines in this file)  to initialize the model with the checkpoint and then perform inference on CityPersons. 

3. Results will be saved at "output/city/val-262.json". Such results is also generated during each evalutation epoch of training. 

3. Change the working directory by "cd eval_city/eval_script". Run the "python3 eval_demo.py" to caculate and output the miss rates of VLPD.

4. Modify the line 509-522 at "eval_MR_multisetup.py" to change the evaluation setting, then repeat (2). You will get the final output as follows: 

    Reasonable+Small+HO+All
    ```
    Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:10000000000.00] ] = 9.41%
    Average Miss Rate  (MR) @ Reasonable_small   [ IoU=0.50      | height=[50:75] | visibility=[0.65:10000000000.00] ] = 10.93%
    Average Miss Rate  (MR) @ Reasonable_occ=heavy [ IoU=0.50      | height=[50:10000000000] | visibility=[0.20:0.65] ] = 34.88%
    Average Miss Rate  (MR) @ All                [ IoU=0.50      | height=[20:10000000000] | visibility=[0.20:10000000000.00] ] = 33.69%
    ```

    Reasonable+Heavy+Partial+Bare
    ```
    Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:10000000000.00] ] = 9.41%
    Average Miss Rate  (MR) @ bare               [ IoU=0.50      | height=[50:10000000000] | visibility=[0.90:10000000000.00] ] = 6.08%
    Average Miss Rate  (MR) @ partial            [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:0.90] ] = 8.78%
    Average Miss Rate  (MR) @ heavy              [ IoU=0.50      | height=[50:10000000000] | visibility=[0.00:0.65] ] = 43.13%
    ```

    Reasonable+Small+Medium+Large
    ```
    Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:10000000000.00] ] = 9.41%
    Average Miss Rate  (MR) @ small              [ IoU=0.50      | height=[50:75] | visibility=[0.65:10000000000.00] ] = 10.93%
    Average Miss Rate  (MR) @ middle             [ IoU=0.50      | height=[75:100] | visibility=[0.65:10000000000.00] ] = 3.61%
    Average Miss Rate  (MR) @ large              [ IoU=0.50      | height=[100:10000000000] | visibility=[0.65:10000000000.00] ] = 5.95%
    ```

    Reasonable+Small+"R+HO"+All
    ```
    Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50      | height=[50:10000000000] | visibility=[0.65:10000000000.00] ] = 9.41%
    Average Miss Rate  (MR) @ Reasonable_small   [ IoU=0.50      | height=[50:75] | visibility=[0.65:10000000000.00] ] = 10.93%
    Average Miss Rate  (MR) @ Reasonable+Heavy   [ IoU=0.50      | height=[50:10000000000] | visibility=[0.20:10000000000.00] ] = 21.74%
    Average Miss Rate  (MR) @ All                [ IoU=0.50      | height=[20:10000000000] | visibility=[0.20:10000000000.00] ] = 33.69%
    ```

## Evaluate the results on Caltech

1. Download the [checkpoint](#evaluate-the-trained-checkpoints) of our VLPD for Caltech.

2. Run "sh test.sh" (please use the **lower** lines in this file) to initialize the model with the checkpoint and then perform inference on Caltech. 

3. Results will be saved at "output/caltech/192". Such results is also generated during each evalutation epoch of training. 

4. Run the Matlab platform, then change its "current folder" to "eval_caltech" of this code, open the code "dbEval.m". 

5. Change the number in line 75 ``exps = exps(1);`` to evaluate on different subsets, e.g., 1 for Reasonable, 2 for All and 9 for Heavy. 

6. Run the code "dbEval.m", then the window of results will occur, and the Miss Rate on the specified subset shows by the top right side.
You will get the final output as follows: 

    ```
    Reasonable: 2.27%
    All: 52.37%
    Heavy: 37.72%
    ```

*← Go back to* [README.md](https://github.com/lmy98129/VLPD)