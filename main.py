import copy
from datetime import datetime
import os
import click
import yaml

from fgnn.launch_run import launch_run
from fgnn.utils.logger import get_logger
from fgnn.utils.utils import load_yaml
from fgnn.utils.grid import create_experiment
from fgnn.utils.run import ParallelRun


OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


@cli.command("grid")
@click.option(
    "--parameters",
    default='parameters/QQB.yaml',
    help="Path to the file containing the parameters for a grid search",
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run the grid in parallel",
)
@click.option(
    "--only_create",
    default=False,
    is_flag=True,
    help="Only create the slurm scripts",
)
def grid(parameters, parallel, only_create=False):
    parameters = load_yaml(parameters)
    run_group = parameters.pop("group")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{run_group}"
    log_folder = os.path.join(OUT_FOLDER, grid_name)
    
    runs_parameters = [
        {"group": run_group, **run_parameters} for run_parameters in create_experiment(parameters)
    ]
    
    os.makedirs(log_folder)
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)

    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        run_name = f"{log_folder}/run_{i}"
        os.makedirs(run_name, exist_ok=True)
        if parallel:
            run = ParallelRun(
                run_parameters,
                multi_gpu=False,
                logger=grid_logger,
                run_name=run_name,
            )
            run.launch(
                only_create=only_create,
                script_args=[
                    "--disable_log_params",
                    "--disable_log_on_file",
                ],
            )
        else:
            grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
            launch_run(run_parameters, run_name=run_name)


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--run_name", default=None, help="Name of the run")
@click.option(
    "--disable_log_params",
    default=False,
    is_flag=True,
    help="Disable Log the parameters",
)
@click.option(
    "--disable_log_on_file", default=False, is_flag=True, help="Disable Log on file"
)
@click.option(
    "--run_name",
    default=None,
    help="Name of the run, if not provided, it will be generated based on the current time",
)
def run(
    parameters,
    run_name=None,
    disable_log_params=False,
    disable_log_on_file=False,
):
    parameters = load_yaml(parameters)
    launch_run(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_on_file,
    )

@cli.command("create")
@click.option("-p", "--parameters", default="parameters/preprocessing/feature_distance.yaml", help="Path to the parameters file")
def create(parameters):
    from fgnn.data.create_water import create_graph_water
    parameters = load_yaml(parameters)
    create_graph_water(parameters=parameters)


if __name__ == "__main__":
    cli()
