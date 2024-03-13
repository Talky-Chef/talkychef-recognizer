import time

import numpy as np
import torch
from sentence_transformers import util


class CommandRecognizer:
    def __init__(self, model=None, arr=None):
        self.model = self.init_model(model)
        self.arr = self.init_array(arr)

    def recognize_command(self, sent):
        start_time = time.time()

        embedding_new = self.model.encode(sent, device="cpu")

        command = "result['command']"
        maxerr = 100
        for row in self.arr:
            center_of_mass = row[0:-1]
            center = center_of_mass.astype(np.float32)
            errors_new = 1 - util.cos_sim(embedding_new, center)
            if errors_new <= maxerr:
                maxerr = errors_new
                if errors_new >= 0.2 and command == "result['command']":
                    command = -1
                elif errors_new < 0.2:
                    command = row[-1]

        print(f"Time taken for recognition: {time.time() - start_time} seconds")
        print(sent, command)
        return command

    def init_model(self, model):
        if model is None:
            path = "resources/model.pth"
            model = torch.load(path, map_location=torch.device("cpu"))
        return model

    def init_array(self, arr):
        if arr is None:
            path = "resources/results.csv"
            arr = np.loadtxt(path, delimiter=',')
        return arr
