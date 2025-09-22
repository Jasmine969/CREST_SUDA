from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import trange
import os
from socket import gethostname

host2path = {
    'LAPTOP-1QA0JPIO': 'F:/intestine_results',
    'DESKTOP-EHK58OI': 'F:/EntericNervousSystem/my_work/results',
    'gpu-server': '/data/zhuhong_codes/EntericNervousSystem/my_work/results'
}
RES_PATH: str = host2path[gethostname()]
paraview.simple._DisableFirstRenderCameraReset()

case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
path = f'{RES_PATH}/{case_name}'
png_folder = 'png-fluid-wall'
if not os.path.exists(f'{path}/{png_folder}'):
    os.mkdir(f'{path}/{png_folder}')
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

clipFluid = Clip(
    registrationName='clipFluid',
    Input=isoVolumeFluid,
    ClipType='Box',
    Invert=True
)
clipFluid.ClipType.Position = [0.0, 0, -0.003]
clipFluid.ClipType.Length = [0.04, 0.003, 0.006]
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
# # ColorBy pressure
# ColorBy(isoVolumeFluidDisplay, ('POINTS', 'v', 'X'))
# get color transfer function/color map for 'v'
vCMap = GetColorTransferFunction('v')
vCMap.RescaleTransferFunction(0, 0.02)
vCMap.ApplyPreset('Plasma (matplotlib)', True)
renderView1.ResetCamera(1)
vBar = GetScalarBar(vCMap, renderView1)
vBar.Title = 'v_mag (m/s)'
vBar.ComponentTitle = ''
vBar.Orientation = 'Horizontal'
vBar.ScalarBarLength = 0.2
vBar.ScalarBarThickness = 20
# modify
vBar.TitleColor = [0.0, 0.0, 0.0]
vBar.LabelColor = [0.0, 0.0, 0.0]
# 自定义位置
vBar.WindowLocation = 'Any Location'
vBar.Position = [0.17, 0.75]
arrowDisplay.SetScalarBarVisibility(renderView1, False)

# get layout
layout1 = GetLayout()
# layout/tab size in pixels
layout1.SetSize(1600, 450)

renderView1.CameraPosition = [0.02, -0.02890155544332871, -9.011154915489726e-05]
renderView1.CameraFocalPoint = [0.02, 0.0011291119793148973, -9.011154915489726e-05]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = 0.020159885748810665

# Solid ====================================
threshSolid = Threshold(
    registrationName='threshSolid',
    Input=reader,  # 如果不指定Input，默认为active source
    Scalars='species',
    ThresholdMethod='Below Lower Threshold',
    LowerThreshold=1.5
)
threshSolidDisplay = Show(threshSolid, renderView1)
threshSolidDisplay.Representation = 'Point Gaussian'
threshSolidDisplay.GaussianRadius = 1e-4  # 与dL保持一致
renderView1.ResetCamera()

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
sphInterpSolid.Source.Scale = [0.04, 0.006, 0.006]
sphInterpSolid.Source.RefinementMode = 'Use cell-size'
sphInterpSolid.Source.CellSize = 5e-5
Hide(threshSolid, renderView1)
sphInterpoSolidDisplay = Show(sphInterpSolid, view=renderView1)
# trace defaults for the display properties.
sphInterpoSolidDisplay.Representation = 'Volume'

isoVolumeSolid = IsoVolume(
    registrationName='IsoVolumeSolid',
    Input=sphInterpSolid,
    InputScalars='Shepard Summation',
    ThresholdRange=[0.7, 100]  # 先设一个比较大的，对于每一帧根据得到的厚度调整
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
clipSolid.ClipType.Position = [0.0, 0, -0.003]
clipSolid.ClipType.Length = [0.04, 0.003, 0.006]
Hide(prog, renderView1)
clipSolidDisplay = Show(clipSolid, renderView1)
HideInteractiveWidgets(proxy=clipSolid)
# ColorBy strain
ColorBy(clipSolidDisplay, 'mystrain')
# get color transfer function/color map for 'c_strainAvg'
strainCMap = GetColorTransferFunction('mystrain')
strainCMap.RescaleTransferFunction(-0.41, 0.2)
strainCMap.RGBPoints = [-0.41, 0.54296875, 0., 0.,
                        0, 1, 1, 1,
                        0.2, 0.2734375, 0.5078125, 0.703125]
strainCMap.ColorSpace = 'RGB'
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
# 自定义位置
strainBar.WindowLocation = 'Any Location'
strainBar.Position = [0.5, 0.75]

renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.OrientationAxesVisibility = 0


def updateSPH():
    sphInterpFluid.DensityArray = 'c_p'
    sphInterpSolid.DensityArray = 'c_p'
    sphInterpFluid.DensityArray = 'c_rho'
    sphInterpSolid.DensityArray = 'c_rho'


for i in trange(251):
    i = 179
    animationScene1.AnimationTime = timesteps[i]
    updateSPH()
    renderView1.Update()
    SaveScreenshot(filename=f'{path}/{png_folder}/wallSI{i:02}.png',
                   viewOrLayout=renderView1, ImageResolution=[1600, 450])
    break
