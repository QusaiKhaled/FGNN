import torch

def get_scheduler(opt, parameters):

    scheduler_name = parameters["name"]
    params = {k: v for k, v in parameters.items() if k != "name"}

    scheduler_class = torch.optim.lr_scheduler.__dict__.get(scheduler_name, None)

    return scheduler_class(
        opt,
        **params
    )