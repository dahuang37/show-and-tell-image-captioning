import os
import torch
import mscoco



inp = "mscoco"

x = getattr(eval(inp), "get_data_loader")

x("msg")