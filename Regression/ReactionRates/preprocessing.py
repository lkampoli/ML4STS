#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

def mk_tree(filename, parent_dir, process, algorithm):

        data = filename[18:36]
        dir  = data[9:14]
        proc = data[15:18]

        print("Dataset: ", data)
        print("Folder: ", dir)
        print("Process: ", proc)
        print("parent_dir: ", parent_dir)

        models  = "models"
        scalers = "scalers"
        figures = "figures"

        model  = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, models)
        scaler = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, scalers)
        figure = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, figures)

        print(model)
        print(scaler)
        print(figure)

        shutil.rmtree(data,   ignore_errors=True)
        shutil.rmtree(model,  ignore_errors=True)
        shutil.rmtree(scaler, ignore_errors=True)
        shutil.rmtree(figure, ignore_errors=True)

        print("Model: ", model)
        print("Scaler: ", scaler)
        print("Figure: ", figure)

        Path(parent_dir+"/"+process+"/"+algorithm+"/"+data).mkdir(parents=True, exist_ok=True)

        os.mkdir(model)
        os.mkdir(scaler)
        os.mkdir(figure)

        print("Directory '%s' created" %models)
        print("Directory '%s' created" %scalers)
        print("Directory '%s' created" %figures)

        return data, dir, proc, model, scaler, figure
