"""Torch with CUDA support bootstrapper.

Reaches into Rhino 3d scripting API to use pip directly to install
torch with cuda support from a custom index url.
Installs all packages into site-packages environment since other
envs under site-env are not suited for complex libs like torch.
Uses scripting progress bar reporter to report install progress.

Reach out for questions to Ehsan Iran-Nejad (ehsan@mcneel.com) or
https://discourse.mcneel.com/u/eirannejad
"""
import threading
from typing import List

# ref necessary rhinocode api
import clr
clr.AddReference("Rhino.Runtime.Code")
clr.AddReference("RhinoCodePlatform.Rhino3D")
from Rhino.Runtime.Code import RhinoCode, ProcessResult, ProgressReport
from Rhino.Runtime.Code.Languages import LanguageSpec
from RhinoCodePlatform.Rhino3D.Languages import RhinoProgressBarRestoreReporter
from Rhino import RhinoApp


# global flag to avoid repeating work
# every time this module is loaded in the same process.
__bootstrapped__ = False


def run_pip(py3, reporter, args, results):
    """PIP installs using given arguments and adds result to the input results list"""
    result = py3.Environs.Shared.PIP(py3.Runtime, args, reporter, lambda _: None)
    reporter.Report(ProgressReport.Complete)    # completes progress and clears Rhino's progressbar
    results.append(result)


def run_pip_async(args): 
    """PIP installs on non-ui thread using given arguments"""
    py3 = RhinoCode.Languages.QueryLatest(LanguageSpec.Python3)
    if not py3:
        raise Exception("Python 3 is somehow not found! Talk to ehsan@mcneel.com")

    # start pip install on a separate thread
    results: List[ProcessResult] = []
    t = threading.Thread(target=run_pip, args=[py3, RhinoProgressBarRestoreReporter(), args, results])
    t.start()

    # pump the ui and wait for install result
    while t.is_alive() and not results:
        RhinoApp.Wait()
    return results[0]


if not __bootstrapped__:
    r = run_pip_async("install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    if r.ExitCode != 0:
        raise Exception(f"Failed installing torch with CUDA support: {r.Output}")

    r = run_pip_async("install diffusers accelerate transformers opencv-python pyOpenSSL controlnet_aux pillow mediapipe numpy timm peft")
    if r.ExitCode != 0:
        raise Exception(f"Failed installing support packages: {r.Output}")

    __bootstrapped__ = True