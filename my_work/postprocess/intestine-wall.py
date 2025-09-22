from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
from socket import gethostname
import os
from multiprocessing import Pool, Manager
from time import time
import gc
import logging
from logging.handlers import QueueHandler, QueueListener

transparent_bg = False
w_img, h_img = 1500, 400


def init_worker(log_queue):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(logging.INFO)


case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
hostname = gethostname()
if hostname == 'DESKTOP-EHK58OI':
    path = f'F:/EntericNervousSystem/my_work/results/{case_name}'
    font_path = 'C:\\Windows\\Fonts\\arial.ttf'
elif hostname == 'gpu-server':
    path = f'/data/zhuhong_codes/EntericNervousSystem/my_work/results/{case_name}'
    font_path = '/usr/share/fonts/truetype/arial/arial.ttf'
else:
    path = f'F:/intestine_results/{case_name}'
    font_path = 'C:\\Windows\\Fonts\\arial.ttf'
if transparent_bg:
    png_folder = 'png-wall_transparent'
else:
    png_folder = 'png-wall'
os.makedirs(f'{path}/{png_folder}', exist_ok=True)
if not os.path.exists(f'{path}/intestine-wall.pvsm'):
    paraview.simple._DisableFirstRenderCameraReset()
    reader = VisItLAMMPSDumpReader(
        registrationName='reader',
        FileName=f'{path}/0to1250000.dump',
        Meshes=['mesh'],
        PointArrays=['c_rho', 'mass',
                     'species', 'x', 'y', 'z']
    )
    tk = GetTimeKeeper()
    timesteps = tk.TimestepValues
    animationScene1 = GetAnimationScene()
    animationScene1.AnimationTime = timesteps[1]
    renderView1 = GetActiveViewOrCreate('RenderView')
    readerDisplay = Show(reader, renderView1)
    readerDisplay.Representation = 'Point Gaussian'
    readerDisplay.GaussianRadius = 1e-4  # ?dL????
    renderView1.ResetCamera()

    # Solid ====================================
    threshSolid = Threshold(
        registrationName='threshSolid',
        Input=reader,  # ?????Input????active source
        Scalars='species',
        ThresholdMethod='Below Lower Threshold',
        LowerThreshold=1.5
    )
    Hide(reader, renderView1)
    threshSolidDisplay = Show(threshSolid, renderView1)
    threshSolidDisplay.Representation = 'Point Gaussian'
    threshSolidDisplay.GaussianRadius = 1e-4  # ?dL????
    renderView1.ResetCamera()

    sphInterpSolid = SPHVolumeInterpolator(
        registrationName='sphInterpSolid',
        Input=threshSolid,
        Source='Bounded Volume',
        DensityArray='c_rho',
        MassArray='mass',
        ExcludedArrays=['species', 'mass', 'x', 'y', 'z',
                        'c_rho']
    )
    sphInterpSolid.Kernel.SpatialStep = 1.7e-4
    sphInterpSolid.Source.Origin = [0, -0.003, -0.003]
    sphInterpSolid.Source.Scale = [0.0398, 0.006, 0.006]
    sphInterpSolid.Source.RefinementMode = 'Use cell-size'
    sphInterpSolid.Source.CellSize = 5e-5
    Hide(threshSolid, renderView1)
    sphInterpoSolidDisplay = Show(sphInterpSolid, view=renderView1)
    # trace defaults for the display properties.
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
        strain = np.load('{path}/interface/strain_tension_1250000.npz')['strain'][id_strain]
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
    clipSolid.ClipType.Position = [0.0, -0.003, -0.003]
    clipSolid.ClipType.Length = [0.0398, 0.006, 0.006]
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
    # strainBar.TitleFontFamily = 'File'
    # strainBar.TitleFontFile = font_path
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
    # white background
    renderView1.UseColorPaletteForBackground = 0
    renderView1.Background = [1.0, 1.0, 1.0]
    renderView1.OrientationAxesVisibility = 0
    SaveState(f'{path}/intestine-wall.pvsm')
    ResetSession()


def process(step):
    me = os.getpid()
    t0_me = time()
    logger = logging.getLogger()
    logger.info(f'Proc {me} starts to process step {step}')
    try:
        LoadState(f'{path}/intestine-wall.pvsm')
        tk = GetTimeKeeper()
        timesteps = tk.TimestepValues
        animationScene1 = GetAnimationScene()
        animationScene1.AnimationTime = timesteps[step]
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.Update()
        SaveScreenshot(filename=f'{path}/{png_folder}/wallSI{step:03}.png',
                       viewOrLayout=renderView1, ImageResolution=[w_img, h_img],
                       TransparentBackground=transparent_bg)
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
    # t0 = time()
    # manager = Manager()
    # log_queue = manager.Queue()
    # file_handler = logging.FileHandler('debug-savestate.log', mode='w', encoding='utf-8')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # listener = QueueListener(log_queue, file_handler)
    # listener.start()
    # t0 = time()
    # with Pool(
    #         processes=6,
    #         initializer=init_worker,
    #         initargs=(log_queue,),
    # ) as pool:
    #     pool.map(process, range(176, 182, 1))
    # listener.stop()
    # file_handler.close()
    # manager.shutdown()
    # print(f'Total time elapsed: {time() - t0} seconds')
    process(176)
