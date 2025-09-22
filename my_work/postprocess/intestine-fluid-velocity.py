from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
import os
from socket import gethostname
import numpy as np
import multiprocessing as mp
from time import time
import gc

print("Rendering backend:", GetRenderView().GetRenderingBackend())

host2path = {
    'LAPTOP-1QA0JPIO': 'F:/intestine_results',
    'DESKTOP-EHK58OI': 'F:/EntericNervousSystem/my_work/results',
    'gpu-server': '/data/zhuhong_codes/EntericNervousSystem/my_work/results'
}
RES_PATH: str = host2path[gethostname()]
paraview.simple._DisableFirstRenderCameraReset()

n_x = 2
case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
path = f'{RES_PATH}/{case_name}'
if not os.path.exists(f'{path}/intestine-fluid-velocity.pvsm'):
    # 读取文件
    reader = VisItLAMMPSDumpReader(
        registrationName='reader',
        FileName=f'{path}/0to1250000.dump',
        Meshes=['mesh'],
        PointArrays=['c_p', 'c_rho', 'mass',
                     'species', 'vx', 'vy', 'vz', 'x', 'y', 'z']
    )
    tk = GetTimeKeeper()
    timesteps = tk.TimestepValues
    animationScene1 = GetAnimationScene()
    animationScene1.AnimationTime = timesteps[1]
    # 显示结果
    renderView1 = GetActiveViewOrCreate('RenderView')
    readerDisplay = Show(reader, renderView1)
    readerDisplay.Representation = 'Point Gaussian'
    readerDisplay.GaussianRadius = 1e-4  # 与dL保持一致
    # 将视图缩放到合适的视角
    renderView1.ResetCamera()

    threshFluid = Threshold(
        registrationName='threshFluid',
        Input=reader,  # 如果不指定Input，默认为active source
        Scalars='species',
        ThresholdMethod='Above Upper Threshold',
        UpperThreshold=1.5
    )
    Hide(reader, renderView1)
    threshFluidDisplay = Show(threshFluid, renderView1)
    threshFluidDisplay.Representation = 'Point Gaussian'
    threshFluidDisplay.GaussianRadius = 1e-4  # 与dL保持一致
    renderView1.ResetCamera(1)

    mergeV = MergeVectorComponents(
        registrationName='mergeV',
        Input=threshFluid,
        XArray='vx', YArray='vy', ZArray='vz',
        OutputVectorName='v'
    )
    Hide(threshFluid, renderView1)
    mergeVDisplay = Show(mergeV, renderView1)
    mergeVDisplay.Representation = 'Point Gaussian'
    mergeVDisplay.GaussianRadius = 1e-4  # 与dL保持一致
    renderView1.ResetCamera(1)

    sphInterpFluid = SPHVolumeInterpolator(
        registrationName='sphInterpFluid',
        Input=mergeV,
        Source='Bounded Volume',
        DensityArray='c_rho',
        MassArray='mass',
        ExcludedArrays=['species', 'mass', 'vx', 'vy', 'vz', 'x', 'y', 'z'],
    )
    sphInterpFluid.Kernel.SpatialStep = 1.3e-4
    l_buf = 0.005
    sphInterpFluid.Source.Origin = [-l_buf, -0.003, -0.003]
    sphInterpFluid.Source.Scale = [0.04 + l_buf * 2, 0.006, 0.006]
    sphInterpFluid.Source.RefinementMode = 'Use cell-size'
    sphInterpFluid.Source.CellSize = 5e-5
    Hide(mergeV, renderView1)
    sphInterpFluidDisplay = Show(sphInterpFluid, view=renderView1)
    sphInterpFluidDisplay.Representation = 'Volume'

    isoVolumeFluid = IsoVolume(
        registrationName='IsoVolumeFluid',
        Input=sphInterpFluid,
        InputScalars='Shepard Summation',
        ThresholdRange=[0.7, 100]
    )
    Hide(sphInterpFluid, renderView1)
    isoVolumeFluidDisplay = Show(isoVolumeFluid, renderView1)
    SaveState(f'{path}/intestine-fluid-velocity.pvsm')
    ResetSession()


def process(frame):
    me = mp.current_process().pid
    start_time = time()
    results = []
    try:
        print(f"Proc {me} starts to process step {frame}")
        LoadState(f'{path}/intestine-fluid-velocity.pvsm')
        tk = GetTimeKeeper()
        timesteps = tk.TimestepValues
        animationScene1 = GetAnimationScene()
        animationScene1.AnimationTime = timesteps[frame]
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.Update()
        isoVolumeFluid = FindSource('IsoVolumeFluid')
        for i_x, x in enumerate([0, 0.04]):
            slice = Slice(
                registrationName='slice',
                Input=isoVolumeFluid,
            )
            slice.SliceType.Origin = [x, 0, 0]
            slice.SliceType.Normal = [1, 0, 0]
            Hide(isoVolumeFluid, renderView1)
            sliceDisplay = Show(slice, renderView1)

            flow = SurfaceFlow(
                registrationName='flow',
                Input=slice,
                SelectInputVectors=['POINTS', 'v']
            )
            flow_array = servermanager.Fetch(flow).GetPointData().GetArray("Surface Flow")
            results.append(flow_array.GetValue(0)*1e9)   # uL/s
            elapsed = time() - start_time
            for obj in ['slice', 'flow']:
                exec(f'Delete({obj})')
                exec(f'del {obj}')
        print(f"Proc {me} finished step {frame}, time elapsed: {elapsed:.2f}s\n")
        for obj in ['renderView1', 'animationScene1']:
            exec(f'Delete({obj})')
            exec(f'del {obj}')
        ResetSession()
    except Exception as e:
        raise RuntimeError(f'Step {frame} failed. Me: {me}. Exception: {e} Obj: {obj}')
    finally:
        gc.collect()
    return results


if __name__ == '__main__':
    n_frame = 250
    flow_rates = np.zeros((n_frame, n_x))
    pool = mp.Pool(processes=5)
    flow_results = []
    for frame in range(1, n_frame + 1):
        flow_results.append(pool.apply_async(process, (frame,)))
    for i, val in enumerate(flow_results):
        flow_rates[i] = val.get()
    pool.close()
    pool.join()
    np.save(f'{path}/flow-rates-paraview', flow_rates)
