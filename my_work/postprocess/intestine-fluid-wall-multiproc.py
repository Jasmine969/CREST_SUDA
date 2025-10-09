from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
from socket import gethostname
import os
from multiprocessing import Pool, Manager
from time import time
import gc
import logging
from logging.handlers import QueueHandler, QueueListener

transparent_bg = True
scale_factor = 1.5
w_img, h_img = int(1500 * scale_factor), int(400 * scale_factor)


def init_worker(log_queue):
    """Initialize the logger"""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(logging.INFO)


case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
host2path = {
    'LAPTOP-1QA0JPIO': 'F:/intestine_results',
    'DESKTOP-EHK58OI': 'F:/EntericNervousSystem/my_work/results',
    'gpu-server': '/data/zhuhong_codes/EntericNervousSystem/my_work/results'
}
hostname = gethostname()
if hostname in host2path:
    RES_PATH: str = host2path[gethostname()]
else:
    RES_PATH = '../my_work/results'
case_path = f'{RES_PATH}/{case_name}'
if transparent_bg:
    png_folder = 'png-fluid-wall_transparent'
    svg_folder = 'svg-fluid-wall_transparent'
else:
    png_folder = 'png-fluid-wall'
    svg_folder = 'svg-fluid-wall'
os.makedirs(f'{case_path}/{png_folder}', exist_ok=True)
os.makedirs(f'{case_path}/{svg_folder}', exist_ok=True)
if not os.path.exists(f'{case_path}/intestine-fluid-wall.pvsm'):
    paraview.simple._DisableFirstRenderCameraReset()
    # read dump file
    reader = VisItLAMMPSDumpReader(
        registrationName='reader',
        FileName=f'{case_path}/0to1250000.dump',
        Meshes=['mesh'],
        PointArrays=['c_p', 'c_rho', 'mass',
                     'species', 'vx', 'vy', 'vz', 'x', 'y', 'z']
    )
    tk = GetTimeKeeper()
    timesteps = tk.TimestepValues
    animationScene1 = GetAnimationScene()
    animationScene1.AnimationTime = timesteps[1]
    renderView1 = GetActiveViewOrCreate('RenderView')
    readerDisplay = Show(reader, renderView1)
    readerDisplay.Representation = 'Point Gaussian'
    readerDisplay.GaussianRadius = 1e-4
    renderView1.ResetCamera()
    # screen out the fluid
    threshFluid = Threshold(
        registrationName='threshFluid',
        Input=reader,
        Scalars='species',
        ThresholdMethod='Above Upper Threshold',
        UpperThreshold=1.5
    )
    Hide(reader, renderView1)
    threshFluidDisplay = Show(threshFluid, renderView1)
    threshFluidDisplay.Representation = 'Point Gaussian'
    threshFluidDisplay.GaussianRadius = 1e-4
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
    mergeVDisplay.GaussianRadius = 1e-4
    renderView1.ResetCamera(1)

    sphInterpFluid = SPHVolumeInterpolator(
        registrationName='sphInterpFluid',
        Input=mergeV,
        Source='Bounded Volume',
        DensityArray='c_rho',
        MassArray='mass',
        ExcludedArrays=['species', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'c_strainAvg'],
    )
    sphInterpFluid.Kernel.SpatialStep = 1.3e-4
    l_buf = 0.005
    sphInterpFluid.Source.Origin = [-l_buf, -0.003, -0.003]
    sphInterpFluid.Source.Scale = [0.0398 + l_buf * 2, 0.006, 0.006]
    sphInterpFluid.Source.RefinementMode = 'Use cell-size'
    sphInterpFluid.Source.CellSize = 5e-5
    Hide(mergeV, renderView1)
    sphInterpFluidDisplay = Show(sphInterpFluid, view=renderView1)
    sphInterpFluidDisplay.Representation = 'Volume'

    isoVolumeFluid = IsoVolume(
        registrationName='isoVolumeFluid',
        Input=sphInterpFluid,
        InputScalars='Shepard Summation',
        ThresholdRange=[0.7, 100]
    )
    Hide(sphInterpFluid, renderView1)
    isoVolumeFluidDisplay = Show(isoVolumeFluid, renderView1)

    clipFluid = Clip(
        registrationName='clipFluid',
        Input=isoVolumeFluid,
        ClipType='Box',
        Invert=True
    )
    clipFluid.ClipType.Position = [0.0, 0, -0.003]
    clipFluid.ClipType.Length = [0.0398, 0.003, 0.006]
    Hide(isoVolumeFluid, renderView1)
    clipFluidDisplay = Show(clipFluid, renderView1)
    HideInteractiveWidgets(proxy=clipFluid)

    calc_vsmooth = Calculator(
        registrationName='calc_vsmooth',
        Input=clipFluid,
        ResultArrayName='vsmooth',
        Function='(v_X^2+v_Y^2+v_Z^2)^0.2'
    )
    arrow = Glyph(
        registrationName='arrow',
        Input=calc_vsmooth,
        GlyphType='Arrow',
        GlyphMode='Uniform Spatial Distribution (Volume Sampling)',
        OrientationArray='v',
        ScaleArray='vsmooth',
        ScaleFactor=0.01,
        MaximumNumberOfSamplePoints=300
    )
    arrowDisplay = Show(arrow, renderView1)
    Hide(clipFluid, renderView1)
    ColorBy(arrowDisplay, ('POINTS', 'v', 'Magnitude'))
    # get color transfer function/color map for 'v'
    vCMap = GetColorTransferFunction('v')
    vCMap.RescaleTransferFunction(0, 0.025)
    vCMap.ApplyPreset('Plasma (matplotlib)', True)
    renderView1.ResetCamera(1)
    vBar = GetScalarBar(vCMap, renderView1)
    vBar.Title = 'v_mag (m/s)'
    vBar.ComponentTitle = ''
    vBar.Orientation = 'Horizontal'
    vBar.ScalarBarLength = 0.2
    vBar.ScalarBarThickness = 20
    vBar.TitleColor = [0.0, 0.0, 0.0]
    vBar.LabelColor = [0.0, 0.0, 0.0]
    vBar.TitleFontSize = 23
    vBar.WindowLocation = 'Any Location'
    vBar.Position = [0.17, 0.75]

    # Solid ====================================
    threshSolid = Threshold(
        registrationName='threshSolid',
        Input=reader,
        Scalars='species',
        ThresholdMethod='Below Lower Threshold',
        LowerThreshold=1.5
    )
    threshSolidDisplay = Show(threshSolid, renderView1)
    threshSolidDisplay.Representation = 'Point Gaussian'
    threshSolidDisplay.GaussianRadius = 1e-4
    renderView1.ResetCamera()

    # SPH interpolation is done to form the continuum, but the resultant strain is not correct.
    # We correct it in the ProgrammableFilter
    sphInterpSolid = SPHVolumeInterpolator(
        registrationName='sphInterpSolid',
        Input=threshSolid,
        Source='Bounded Volume',
        DensityArray='c_rho',
        MassArray='mass',
        ExcludedArrays=['species', 'mass', 'x', 'y', 'z',
                        'vx', 'vy', 'vz', 'c_p', 'c_rho']
    )
    sphInterpSolid.Kernel.SpatialStep = 1.7e-4
    sphInterpSolid.Source.Origin = [0, -0.003, -0.003]
    sphInterpSolid.Source.Scale = [0.0398, 0.006, 0.006]
    sphInterpSolid.Source.RefinementMode = 'Use cell-size'
    sphInterpSolid.Source.CellSize = 5e-5
    Hide(threshSolid, renderView1)
    sphInterpoSolidDisplay = Show(sphInterpSolid, view=renderView1)
    sphInterpoSolidDisplay.Representation = 'Volume'

    isoVolumeSolid = IsoVolume(
        registrationName='isoVolumeSolid',
        Input=sphInterpSolid,
        InputScalars='Shepard Summation',
        ThresholdRange=[0.7, 100]
    )
    Hide(sphInterpSolid, renderView1)
    isoVolumeSolidDisplay = Show(isoVolumeSolid, renderView1)

    prog = ProgrammableFilter(
        registrationName='prog',
        Input=isoVolumeSolid,
        Script=f"""
        def get_frame(inp, step_interval):
            import io
            import sys
            output_buffer = io.StringIO()
            # redirection
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            # print info to outbuffer
            print(inp.GetInformation())
            # restore stdout
            sys.stdout = original_stdout
            captured_output = output_buffer.getvalue()
            output_buffer.close()
            for each in captured_output.splitlines():
                if each.strip().startswith('DATA_TIME_STEP'):
                    return int(float(each.split(':')[1])) // step_interval


        import numpy as np
        from scipy.interpolate import PchipInterpolator as pchip
        from vtk.util.numpy_support import vtk_to_numpy
        input0 = inputs[0]
        frame = get_frame(input0, 5000)
        id_strain = int(frame * 100 - 1)
        strain = np.load('{case_path}/interface/strain_tension_1250000.npz')['strain'][id_strain]
        interp = pchip(2e-4 * np.arange(200), strain)
        x = vtk_to_numpy(input0.GetBlock(0).GetPoints().GetData())[:,0]
        output.PointData.append(interp(x), "mystrain")
        """,
        CopyArrays=True
    )
    Hide(isoVolumeSolid, renderView1)
    progDisplay = Show(prog, view=renderView1)

    clipSolid = Clip(
        registrationName='clipSolid',
        Input=prog,
        ClipType='Box',
        Invert=True
    )
    clipSolid.ClipType.Position = [0.0, 0, -0.003]
    clipSolid.ClipType.Length = [0.0398, 0.003, 0.006]
    Hide(prog, renderView1)
    clipSolidDisplay = Show(clipSolid, renderView1)
    HideInteractiveWidgets(proxy=clipSolid)
    # ColorBy strain
    ColorBy(clipSolidDisplay, 'mystrain')
    # get color transfer function/color map for 'c_strainAvg'
    strainCMap = GetColorTransferFunction('mystrain')
    strainCMap.RescaleTransferFunction(-0.41, 0.2)
    strainCMap.ColorSpace = 'RGB'
    strainCMap.RGBPoints = [-0.41, 0.54296875, 0., 0.,
                            0, 1, 1, 1,
                            0.2, 0.2734375, 0.5078125, 0.703125]
    renderView1.ResetCamera(1)
    strainBar = GetScalarBar(strainCMap, renderView1)
    strainBar.Title = 'Strain'
    strainBar.ComponentTitle = ''
    strainBar.Orientation = 'Horizontal'
    strainBar.ScalarBarLength = 0.2
    strainBar.ScalarBarThickness = 20
    strainBar.TitleColor = [0.0, 0.0, 0.0]
    strainBar.LabelColor = [0.0, 0.0, 0.0]
    strainBar.TitleFontSize = 23
    strainBar.WindowLocation = 'Any Location'
    strainBar.Position = [0.5, 0.75]

    # get layout
    layout1 = GetLayout()
    if not layout1:
        layout1 = CreateLayout(name="MainLayout")
        AssignViewToLayout(view=renderView1, layout=layout1)
    # layout/tab size in pixels
    layout1.SetSize(w_img, h_img)

    renderView1.CameraPosition = [0.02, -0.021, 0]
    renderView1.CameraFocalPoint = [0.02, 0.00113, 0]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 0.02
    # Set the background color to white
    renderView1.UseColorPaletteForBackground = 0
    renderView1.Background = [1.0, 1.0, 1.0]
    renderView1.OrientationAxesVisibility = 0
    SaveState(f'{case_path}/intestine-fluid-wall.pvsm')
    ResetSession()


def process(step):
    me = os.getpid()
    t0_me = time()
    logger = logging.getLogger()
    logger.info(f'Proc {me} starts to process step {step}')
    try:
        LoadState(f'{case_path}/intestine-fluid-wall.pvsm')
        tk = GetTimeKeeper()
        timesteps = tk.TimestepValues
        animationScene1 = GetAnimationScene()
        animationScene1.AnimationTime = timesteps[step]
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.Update()
        SaveScreenshot(filename=f'{case_path}/{png_folder}/wallSI{step:03}.png',
                       viewOrLayout=renderView1, ImageResolution=[w_img, h_img],
                       TransparentBackground=transparent_bg, CompressionLevel=0)
        if step == 6: # export svg only once because only colorbar is svg
            ExportView(filename=f'{case_path}/{svg_folder}/wallSI{step:03}.svg',
               view=renderView1)
        logger.info(f'{me} finished timestep {step}, time elapsed: {time() - t0_me} s')
        for obj in ['renderView1', 'animationScene1']:
            exec(f'Delete({obj})')
            exec(f'del {obj}')
        ResetSession()
    except Exception as e:
        logger.error(f'Step {step} failed. Me: {me}. Exception: {e} Obj: {obj}', exc_info=True)
        raise RuntimeError
    finally:
        gc.collect()


if __name__ == '__main__':
    t0 = time()
    manager = Manager()
    log_queue = manager.Queue()
    file_handler = logging.FileHandler('debug-savestate.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    t0 = time()
    with Pool(
            processes=3,
            initializer=init_worker,
            initargs=(log_queue,),
    ) as pool:
        pool.map(process, range(1, 251, 1))
        # pool.map(process, [6, 115, 200])
    listener.stop()
    file_handler.close()
    manager.shutdown()
    print(f'Total time elapsed: {time() - t0} seconds')
    # process(1)
