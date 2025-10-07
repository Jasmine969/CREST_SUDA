"""
The strains obtained by reading the dump file and calculating mean(sqrt((y-yc)^2+(z-zc)^2))
 often do not match with those of interface file and c_strainAvg
 because the former are those at a single timestep while the latter are averaged every Ncallback timesteps.
 When strains are plotted vs. force or velocities, the former can be used because forces and velocities
 are also for a single timestep. If you want the former, just run the func extract_ring_strain,
 the results are saved in all_rings_strain.xlsx,
 with the compensation of only 251 timesteps (the latter has 25000 timesteps).
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from utils.result_path import RES_PATH
import os
import pandas as pd
from ovito.io import import_file
import ovito.modifiers as mod

Ncallback_lmp = 50
cases_all = [
    'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain',
    'rheo_bond2_angle-F100-chyme-noICC-30w-ringstrain',
    'rheo_bond2_angle-F100-chymepower-noICC-30w-ringstrain'
]
labels_all = ['Krebs solution', 'Liquid digesta', 'Whole digesta']
current_case = cases_all[0]
case_path = f'{RES_PATH}/{current_case}'
font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
r_si = 2e-3
l_si = 0.04
n_yz = 63
n_rings = 200
n_inlet = 3870
dt_lmp = 2e-5
m0 = 8.7153178088417643497e-09  # kg
rho = 993
dL = 2e-4
h = 1.6 * dL


def flow_rate(filename, ax=None):
    from utils.mathfunc import slope_estimate
    import yaml
    # with open(f'{RES_PATH}/default/in_out_flow_baseline_20w.yml') as base:
    #     data_base = yaml.safe_load(base)
    # data_base = pd.DataFrame(data_base).to_numpy().astype(float)
    # data_base[:, 0] *= 5e-5
    with open(f'{case_path}/{filename}') as f:
        data = yaml.safe_load(f)
    data = pd.DataFrame(data).to_numpy().astype(float)
    data[:, 0] = (data[:, 0] * 2e-5).round(3)  # second
    dt = data[1, 0] - data[0, 0]
    _, inflow_rate, _, _ = slope_estimate(
        data[:, 0], data[:, 1], scale=m0 / rho * 1e9,
        diff_dx=dt, res_dx=dt, orig_frac=0., deri_frac=0.004)
    _, outflow_rate, _, _ = slope_estimate(
        data[:, 0], data[:, 2], scale=m0 / rho * 1e9,
        diff_dx=dt, res_dx=dt, orig_frac=0., deri_frac=0.004)

    rates = np.c_[data[:, 0], inflow_rate, outflow_rate]

    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    ax.plot(rates[:, 0], rates[:, 1])
    ax.plot(rates[:, 0], rates[:, 2])
    # plt.plot(rates_base[:, 0], rates_base[:, 1], '--')
    # plt.plot(rates_base[:, 0], rates_base[:, 2], '--')
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlim([0, 25])
    if external_ax:
        return
    ax.set_xlabel('Time (s)', fontdict=font_label)
    ax.set_ylabel('Flow rate (μL/s)', fontdict=font_label)

    res = np.load(f'{case_path}/flow-rates-paraview.npy')
    t_res = np.arange(0.1, 25.1, 0.1)
    ax.plot(t_res, -res[:, 0])
    ax.plot(t_res, res[:, -2])

    plt.tight_layout()
    plt.show()


def flow_rate_paraview(ax=None):
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    res = np.load(f'{case_path}/flow-rates-paraview.npy')
    t_res = np.arange(0.1, 25.1, 0.1)
    ax.plot(t_res, -res[:, 0])
    ax.plot(t_res, res[:, -2])
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlim([0, 25])
    if external_ax:
        return
    ax.set_xlabel('Time (s)', fontdict=font_label)
    ax.set_ylabel('Flow rate (μL/s)', fontdict=font_label)
    plt.tight_layout()
    plt.show()


def flow_rate_map(ax=None, hlines=None):
    """
    Plot the flow rates on the spatiotemporal map.
    One should first run my_work/postprocess/intestine-fluid-velocity.py to obtain flow-rates-paraview.npy
    """
    from utils.id2x import x2ringID
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    res = np.load(f'{case_path}/flow-rates-paraview.npy')
    im = ax.imshow(res, aspect='auto', cmap='PuOr', vmin=-50, vmax=50)
    cb = plt.colorbar(im, location='bottom')
    cb.set_label('Flow rate (μL/s)', fontdict=font_label)
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels * 10
    if res.shape[0] == 250:
        yticks = yticks - 1
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax.set_ylim([yticks[-1], yticks[0]])
    if hlines:
        if res.shape[0] == 250:
            ax.hlines(np.array(hlines) * 10 - 1, xmin=0, xmax=199, color='k', ls='--')
        else:
            ax.hlines(np.array(hlines) * 10, xmin=0, xmax=199, color='k', ls='--')
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels, fontdict=font_ticks)
    plt.yticks(yticks, ytick_labels, fontdict=font_ticks)
    plt.ylabel('Time (s)', fontdict=font_label)
    plt.xlabel('x (mm)', fontdict=font_label)
    plt.tight_layout()
    plt.show()


def outflow_compare(file_list=None):
    plt.rc('font', **font_ticks)
    import yaml
    if file_list is None:
        file_list = ['in_out_flow_0to1250000.yml'] * 3
    for label, case_name, filename in zip(labels_all, cases_all, file_list):
        with open(f'{RES_PATH}/{case_name}/{filename}', 'r') as f:
            data = yaml.safe_load(f)
        data = pd.DataFrame(data).to_numpy().astype(float)
        data[:, 0] *= 2e-5  # second
        outflow = (data[:, 2] - data[0, 2]) * m0 / rho * 1e9
        plt.plot(data[:, 0], outflow, label=str(label))
    plt.legend(loc='best')
    plt.xlabel('Time (s)', fontdict=font_label)
    plt.ylabel('Outflow (μL)', fontdict=font_label)
    plt.tight_layout()
    plt.show()


def outflow_compare_paraview():
    from utils.mathfunc import time_integrate

    plt.rc('font', **font_ticks)
    t = np.arange(1, 251) * 0.1
    for label, case_name in zip(labels_all, cases_all):
        data = np.load(f'{RES_PATH}/{case_name}/flow-rates-paraview.npy')
        # outflow = time_integrate(t, data[:, -1])
        plt.plot(t, data[:, -2], label=label)
    plt.legend(loc='best')
    plt.xlabel('Time (s)', fontdict=font_label)
    plt.ylabel('Outflow (μL)', fontdict=font_label)
    plt.tight_layout()
    plt.show()


def intraluminal_pressure(dump_name):
    import pickle
    if not os.path.exists(f'{case_path}/pressure.pkl'):
        from utils.sph_interpolate import extract_property
        pipeline = import_file(f'{case_path}/{dump_name}', sort_particles=True)
        n_sample = 50
        xs = np.linspace(0, l_si - dL, n_sample)[:, np.newaxis]
        xyz = np.hstack((xs, np.zeros((n_sample, 2))))
        pressure_sample = extract_property(
            pipeline,
            names_prop=['c_p'],
            xyz=xyz,
            cutoff=h
        )[0]
        pressure_sample /= 0.01 * 9.81 * rho  # cmH2O
        with open(f'{case_path}/pressure.pkl', 'wb') as pf:
            pickle.dump(pressure_sample, pf)
        ims = plt.imshow(pressure_sample.T, aspect='auto')
        plt.colorbar(ims, pad=0.07, shrink=0.7, anchor=(0.0, 0.9))
        plt.xlabel('Time')
        plt.ylabel('Intestine length')
        plt.show()
    else:
        with open(f'{case_path}/pressure.pkl', 'rb') as pf:
            pressure_sample = pickle.load(pf)
        plt.figure(1)
        ims = plt.imshow(pressure_sample.T, aspect='auto')
        plt.colorbar(ims, pad=0.07, shrink=0.7, anchor=(0.0, 0.9))
        plt.xlabel('Time')
        plt.ylabel('Intestine length')
        time = np.arange(0, 25.01, 0.1)
        pressure_diff = pressure_sample[:, -1] - pressure_sample[:, 0]
        plt.figure(2)
        plt.plot(time, pressure_diff)
        plt.xlabel('Time')
        plt.ylabel('Pressure difference (cmH2O)')
        plt.show()


def extract_force_strain_atom(ringID=116, atomID=11238):
    """
    For a single atom, extract the following forces: active force, gravity, bath force, FSI force,
    restoring, viscoelastic force,
    and the local strain.
    Only do extraction without plotting.
    """
    from utils.mathfunc import proj

    if_print = False
    atomID_min = n_inlet + ringID * n_yz + 1
    atomID_max = n_inlet + (ringID + 1) * n_yz
    assert atomID_min <= atomID <= atomID_max

    pipeline = import_file(f'{case_path}/0to1250000.dump', sort_particles=True)
    pipeline.modifiers.extend([
        mod.ExpressionSelectionModifier(
            expression=f'ParticleIdentifier<{atomID_min} || ParticleIdentifier>{atomID_max}'),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])
    pipeline_bond = import_file(f'{case_path}/Fbond.dump', sort_particles=True)
    pipeline_bond.modifiers.extend([
        mod.ExpressionSelectionModifier(
            expression=f'ParticleIdentifier!={atomID}'),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])
    n_frames = pipeline.source.num_frames
    df_force = pd.DataFrame(
        np.zeros((n_frames, 18)),
        columns=['strain_local',
                 'gravity-in', 'gravity-z',
                 'active-in', 'active-y', 'active-z',
                 'bath-in', 'bath-y', 'bath-z',
                 'FSI-in', 'FSI-y', 'FSI-z',
                 'restoring-in', 'restoring-y', 'restoring-z',
                 'viscoelastic-in', 'viscoelastic-y', 'viscoelastic-z'])
    yz_all = np.zeros((n_frames, n_yz, 2))
    center_all = np.zeros((n_frames, 2))
    for frame in trange(n_frames):
        data = pipeline.compute(frame)
        ys: np.ndarray = data.particles['Position.Y']
        zs: np.ndarray = data.particles['Position.Z']
        yz_all[frame, :, 0] = ys
        yz_all[frame, :, 1] = zs
        center_y = ys.mean()
        center_z = zs.mean()
        center_all[frame, :] = [center_y, center_z]

        pipeline.modifiers.extend([
            mod.ExpressionSelectionModifier(
                expression=f'ParticleIdentifier!={atomID}'),
            mod.DeleteSelectedModifier(operate_on={'particles'})
        ])
        data = pipeline.compute(frame)
        y = data.particles['Position.Y'].item()
        z = data.particles['Position.Z'].item()
        unit_scale = 1e6
        unit_name = 'μN'

        # gravity
        f_grav = -9.81 * m0
        df_force.loc[frame, 'gravity-z'] = f_grav
        if if_print:
            print(f'G={f_grav * unit_scale:.1f} {unit_name}')

        # active force
        Fay = data.particles['f_f_active[2]'].array.item()
        Faz = data.particles['f_f_active[3]'].array.item()
        df_force.loc[frame, 'active-y'] = Fay
        df_force.loc[frame, 'active-z'] = Faz
        Fa_mag = np.sqrt(Fay ** 2 + Faz ** 2).item()
        if if_print:
            print(f'F_active={Fa_mag * unit_scale: .1f} {unit_name}')

        # bath force
        fy_bath = data.particles['v_fy_bath'].array.item()
        fz_bath = data.particles['v_fz_bath'].array.item()
        df_force.loc[frame, 'bath-y'] = fy_bath
        df_force.loc[frame, 'bath-z'] = fz_bath
        F_bath_mag = np.sqrt(fy_bath ** 2 + fz_bath ** 2).item()
        if if_print:
            print(f'F_bath={F_bath_mag * unit_scale: .1f} {unit_name}')

        # viscoelastic force
        data_bond = pipeline_bond.compute(frame)
        fy_bond = data_bond.particles['Force.Y'].item()
        fz_bond = data_bond.particles['Force.Z'].item()
        df_force.loc[frame, 'viscoelastic-y'] = fy_bond
        df_force.loc[frame, 'viscoelastic-z'] = fz_bond
        F_bond_mag = np.sqrt(fy_bond ** 2 + fz_bond ** 2).item()
        if if_print:
            print(f'F_viscoelastic={F_bond_mag * unit_scale: .1f} {unit_name}')

        # self-spring restoring force
        fy_restore = data.particles['v_fry'].array.item()
        fz_restore = data.particles['v_frz'].array.item()
        df_force.loc[frame, 'restoring-y'] = fy_restore
        df_force.loc[frame, 'restoring-z'] = fz_restore
        F_restore_mag = np.sqrt(fy_restore ** 2 + fz_restore ** 2).item()
        if if_print:
            print(f'F_restore={F_restore_mag * unit_scale: .1f} {unit_name}')

        # FSI force
        fy_FSI = data.particles['Force.Y'].item() - (fy_bond + fy_bath + fy_restore + Fay)
        fz_FSI = data.particles['Force.Z'].item() - (fz_bond + fz_bath + fz_restore + Faz + f_grav)
        df_force.loc[frame, 'FSI-y'] = fy_FSI
        df_force.loc[frame, 'FSI-z'] = fz_FSI
        F_FSI_mag = np.sqrt(fy_FSI ** 2 + fz_FSI ** 2).item()
        if if_print:
            print(f'F_FSI={F_FSI_mag * unit_scale: .1f} {unit_name}')

        strain_local = (((y - center_y) ** 2 + (z - center_z) ** 2) ** 0.5 - r_si) / r_si
        df_force.loc[frame, 'strain_local'] = strain_local
        # project forces onto the inward normal
        # assume inward direction as positive
        vec_pt_center = np.array([center_y, center_z]) - np.array([y, z])
        df_force.loc[frame, 'gravity-in'] = proj(
            np.array([0, f_grav]), vec_pt_center) * unit_scale
        df_force.loc[frame, 'active-in'] = proj(
            np.array([Fay, Faz]), vec_pt_center) * unit_scale
        df_force.loc[frame, 'bath-in'] = proj(
            np.array([fy_bath, fz_bath]), vec_pt_center) * unit_scale
        df_force.loc[frame, 'FSI-in'] = proj(
            np.array([fy_FSI, fz_FSI]), vec_pt_center) * unit_scale
        df_force.loc[frame, 'restoring-in'] = proj(
            np.array([fy_restore, fz_restore]), vec_pt_center) * unit_scale
        df_force.loc[frame, 'viscoelastic-in'] = proj(
            np.array([fy_bond, fz_bond]), vec_pt_center) * unit_scale
        # remove the last two modifiers to get yz and center in the next frame
        pipeline.modifiers.pop()
        pipeline.modifiers.pop()
    writer = pd.ExcelWriter(f'{case_path}/force-ring{ringID}-atom{atomID}.xlsx')
    df_force.to_excel(writer, index=False)
    writer.close()
    np.savez(f'{case_path}/yz-center-ring{ringID}', yz=yz_all, center=center_all)


def draw_atom_force_arrow(frame, ringID=116, atomID=11238,
                          ax=None, df_force=None, yz_and_center=None,
                          if_annotate=True, spine=True, transparent_bg=False):
    """
    One should first run extract_force_strain_atom.
    If all the frames are to be drawn, the user can load file outside this function
    and pass it via df_force and yz_and_center to avoid repeated loading;
    else, if only a few frames are drawn, df_force and yz_and_center are loaded inside this function.
    """
    from my_work.create_geometry.my_geometry import CylinderSide
    import matplotlib as mpl
    top_annotate = 0.0021
    hspace_annotate = 0.0005
    left_annotate = 0.0024
    len_arrow = 0.0015
    right_annotate = left_annotate + len_arrow
    circle_area = 150
    png_folder = f'png-force_arrow-ring{ringID}-atom{atomID}'
    os.makedirs(f'{case_path}/{png_folder}', exist_ok=True)
    id_yz = (atomID - 1 - n_inlet) % n_yz  # id on the current ring
    si = CylinderSide(r=r_si, l_axis=0, dl=dL, axis='x', print_log=False)
    if df_force is None:
        df_force = pd.read_excel(f'{case_path}/force-ring{ringID}-atom{atomID}.xlsx')
    df_force['gravity-y'] = 0
    if yz_and_center is None:
        yz_and_center = np.load(f'{case_path}/yz-center-ring{ringID}.npz')
    yz_all, center_all = yz_and_center['yz'], yz_and_center['center']
    if ax is None:
        mpl.rcParams['svg.fonttype'] = 'none'
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
        external_ax = False
    else:
        external_ax = True
    ax.scatter(si.ys, si.zs, circle_area, ls='--', fc='w', ec='k')
    ax.scatter(yz_all[frame, :, 0], yz_all[frame, :, 1], circle_area, fc='C0', ec='k', alpha=0.5)
    ax.scatter(center_all[frame, 0], center_all[frame, 1], 30, color='k')
    ax.text(center_all[frame, 0], center_all[frame, 1] + 1e-4, 'Center', fontdict=font_ticks, ha='center')
    if if_annotate:
        ax.annotate('Resting state', (-0.0016, -0.0013), (-0.0022, -0.0023),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='k', ls='--'),
                    horizontalalignment='center', backgroundcolor='w', fontproperties=font_ticks)
    unit_scale = 1e6
    unit_name = 'μN'
    k_norm = 3e-3
    y, z = yz_all[frame, id_yz, :]
    force_now = df_force.loc[frame]
    name_style_height = [('Gravity', 'C0', 0),
                         ('Active', 'C1', 1),
                         ('Bath', 'C2', 2),
                         ('Viscoelastic', 'C3', 3),
                         ('Restoring', 'C4', 4),
                         ('FSI', 'C5', 5)]
    for name, style, height in name_style_height:
        name_ = name if name == 'FSI' else name.lower()
        ax.annotate(
            '', (y, z),
            (y + force_now[f'{name_}-y'] / k_norm, z + force_now[f'{name_}-z'] / k_norm),
            arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0, color=style, lw=3),
            horizontalalignment='center', backgroundcolor='w', fontproperties=font_ticks
        )
        if if_annotate:
            ax.text(left_annotate, top_annotate - height * hspace_annotate + 1e-4,
                    name, fontdict=font_ticks)
            ax.annotate(
                '', (left_annotate, top_annotate - height * hspace_annotate),
                (right_annotate, top_annotate - height * hspace_annotate),
                arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0, color=style, lw=3),
                horizontalalignment='center', backgroundcolor='w', fontproperties=font_ticks
            )
    # scale bar
    if if_annotate:
        ax.text(0.0027, top_annotate - 5 * hspace_annotate - 3e-4,
                f'{len_arrow * k_norm * unit_scale} {unit_name}', fontdict=font_ticks)

    ax.axis('scaled')
    ax.set_xlim([-0.003, 0.004])
    ax.set_ylim([-0.0025, 0.0025])
    if spine:
        xticklabels = np.arange(-3, 5)
        xticks = xticklabels / 1000
        yticklabels = np.arange(-2, 3)
        yticks = yticklabels / 1000
        if external_ax:
            return xticks, xticklabels, yticks, yticklabels
        ax.set_xticks(xticks, xticklabels, fontdict=font_ticks)
        ax.set_yticks(yticks, yticklabels, fontdict=font_ticks)
        ax.set_xlabel('y (mm)', font=font_label)
        ax.set_ylabel('z (mm)', font=font_label)
        ax.set_title(f'Time={frame * 0.1: .1f} sec', fontdict=font_label)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if not external_ax:
        plt.gcf().savefig(f'{case_path}/{png_folder}/{frame: 03}.png', transparent=transparent_bg)
        plt.gcf().savefig(f'{case_path}/{png_folder}/{frame: 03}.svg', transparent=transparent_bg)
        plt.show()


def plot_atom_force_strain(frame, ringID=116, atomID=11238, ax=None, df_force=None):
    """
    One should first run extract_force_strain_atom.
    If all the frames are to be drawn, the user can load file outside this function
    and pass it via df_force to avoid repeated loading;
    else, if only a few frames are drawn, df_force is loaded inside this function.
    """
    if df_force is None:
        df_force = pd.read_excel(f'{case_path}/force-ring{ringID}-atom{atomID}.xlsx')
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
        external_ax = False
    else:
        external_ax = True
    # assume inward direction as positive
    name_style = [('Gravity', 'C0o-'),
                  ('Active', 'C1v-'),
                  ('Bath', 'C2>-'),
                  ('Viscoelastic', 'C3*-'),
                  ('Restoring', 'C4d-'),
                  ('FSI', 'C5s-')]
    for name, style in name_style:
        ax.plot(df_force.loc[0:frame, 'strain_local'],
                df_force.loc[0:frame, f"{name.lower() if name != 'FSI' else name}-in"],
                style, label=name)
    ax.set_xlim([-0.35, 0.19])
    ax.set_ylim([-7.0, 10.5])
    if not external_ax:
        ax.set_xlabel('Local strain', fontdict=font_label)
        ax.set_ylabel('Inward force (μN)', fontdict=font_label)
        ax.legend(loc='upper right', ncol=2)
        plt.show()


def combine_atom_force_strain_arrow(ringID=116, atomID=11238):
    """
    One should first run extract_force_strain_atom
    Combine func draw_atom_force_arrow and func plot_atom_force_strain in a figure.
    To make SV1-force-strain-ring?-atom?.mp4
    """
    os.makedirs(f'{case_path}/png-force-strain-ring{ringID}-atom{atomID}', exist_ok=True)
    df_force = pd.read_excel(f'{case_path}/force-ring{ringID}-atom{atomID}.xlsx')
    yz_and_center = np.load(f'{case_path}/yz-center-ring{ringID}.npz')
    plt.rc('font', **font_ticks)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for frame in trange(df_force.shape[0]):
        xticks, xticklabels, yticks, yticklabels = draw_atom_force_arrow(frame, ax[0], df_force, yz_and_center)
        plot_atom_force_strain(frame, ax[1], df_force)
        ax[0].set_xticks(xticks)
        ax[0].set_xticklabels(xticklabels)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(yticklabels)
        ax[0].set_xlabel('y (mm)', font=font_label)
        ax[0].set_ylabel('z (mm)', font=font_label)
        ax[0].set_title(f'Time={frame * 0.1: .1f} sec', fontdict=font_label)
        ax[1].set_xlabel('Local strain', fontdict=font_label)
        ax[1].set_ylabel('Inward force (μN)', fontdict=font_label)
        ax[1].legend(loc='upper right', ncol=2, prop=font_ticks)
        plt.subplots_adjust(
            hspace=0.2, wspace=0.15,
            top=0.925, bottom=0.122,
            left=0.045, right=0.981
        )
        fig.savefig(f'{case_path}/png-force-strain-ring{ringID}-atom{atomID}/force-{frame:03}.png')
        ax[0].cla()
        ax[1].cla()
        # plt.show()


def force_time_atom(ringID=116, atomID=11238, ax_force=None, ax_strain=None):
    """
    One should first run extract_force_strain_atom
    Plot force-time and strain_local-time simultaneously.
    Ax_force and ax_strain had better share the time-axis.
    """
    assert not (bool(ax_force) ^ bool(ax_strain))
    plt.rc('font', **font_ticks)
    df_force = pd.read_excel(f'{case_path}/force-ring{ringID}-atom{atomID}.xlsx')
    columns_want = ['strain_local'] + [each for each in df_force.columns if '-in' in each]
    df_force = df_force[columns_want]
    columns = dict()
    for each in df_force.columns:
        if each != 'FSI-in':
            columns[each] = each.capitalize().split('-')[0]
        else:
            columns[each] = each.split('-')[0]
    df_force = df_force.rename(columns=columns)
    df_force['Time'] = np.arange(df_force.shape[0]) * 0.1
    styles = {
        'Gravity': 'C0-', 'Active': 'C1-',
        'Bath': 'C2-', 'FSI': 'C5-',
        'Restoring': 'C4-', 'Viscoelastic': 'C3-'
    }
    if ax_force is None:
        fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(10, 5))
        ax_force, ax_strain = ax
        external_ax = False
    else:
        external_ax = True
    df_force.plot(
        x='Time', y=styles.keys(),
        ax=ax_force,
        style=styles,
        xlim=[0, 25], xticks=np.arange(0, 26, 5),
        legend=False, lw=2
    )
    # ax_force.legend(loc='best', ncol=2, prop=font_label)
    df_force.plot(
        x='Time', y='Strain_local',
        ax=ax_strain,
        style='k',
        lw=2, legend=False
    )
    if not external_ax:
        ax_force.set_ylabel('Inward force (μN)', fontdict=font_label)
        ax_strain.set_ylabel('Local strain', fontdict=font_label)
        ax_strain.set_xlabel('Time (s)', fontdict=font_label)
        plt.gcf().savefig(f'{case_path}/force-time-atom{atomID}.png')
        plt.show()


def force_time_atom_paper():
    """
    Plot Fig. 2c
    """
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rc('font', **font_ticks)
    fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(13, 5))
    force_time_atom(ax_force=ax[0], ax_strain=ax[1])
    ax[0].set_ylabel('Inward force (μN)', fontdict=font_label)
    ax[1].set_ylabel('Local strain', fontdict=font_label)
    ax[1].set_xlabel('Time (s)', fontdict=font_label)
    for ax_ in ax:
        ymin, ymax = ax_.get_ylim()
        ax_.vlines([8.0, 8.8, 21.7], ymin, ymax, colors='gray', ls='--')
        ax_.set_ylim([ymin, ymax])
    ax[1].axhline(0, color='gray', ls='--')
    fig.savefig(f'{case_path}/force-time-ring116-atom11238.png')
    fig.savefig(f'{case_path}/force-time-ring116-atom11238.svg', transparent=True)
    plt.show()


def extract_force_ring(ringID):
    """
    Almost the same as extract_force_strain_atom,
    but average the forces of a ring.
    The strain is not obtained here, but by running extract_all_rings_strain
    """
    from utils.mathfunc import proj
    plt.rc('font', **font_ticks)
    atomID_min = n_inlet + ringID * n_yz + 1
    atomID_max = n_inlet + (ringID + 1) * n_yz
    pipeline = import_file(f'{case_path}/0to1250000.dump', sort_particles=True)
    pipeline.modifiers.extend([
        mod.ExpressionSelectionModifier(
            expression=f'ParticleIdentifier<{atomID_min} || ParticleIdentifier>{atomID_max}'),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])
    pipeline_bond = import_file(f'{case_path}/Fbond.dump', sort_particles=True)
    pipeline_bond.modifiers.extend([
        mod.ExpressionSelectionModifier(
            expression=f'ParticleIdentifier<{atomID_min} || ParticleIdentifier>{atomID_max}'),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])

    n_frames = pipeline.source.num_frames
    df_force = pd.DataFrame(
        np.zeros((n_frames, 6)),
        columns=['gravity', 'active', 'bath',
                 'FSI', 'restoring', 'viscoelastic'])
    unit_scale = 1e6
    for frame in trange(n_frames):
        data = pipeline.compute(frame)
        yz = data.particles['Position'].array[:, [1, 2]]
        center_yz = yz.mean(axis=0, keepdims=True)
        vec_pt_center = center_yz - yz  # (63,2)
        f_grav = -9.81 * m0
        df_force.loc[frame, 'gravity'] = np.mean(proj(
            np.tile([0, f_grav], (n_yz, 1)), vec_pt_center)) * unit_scale
        Fay = data.particles['f_f_active[2]'].array
        Faz = data.particles['f_f_active[3]'].array
        df_force.loc[frame, 'active'] = np.mean(proj(
            np.c_[Fay, Faz], vec_pt_center)) * unit_scale
        fy_bath = data.particles['v_fy_bath'].array
        fz_bath = data.particles['v_fz_bath'].array
        df_force.loc[frame, 'bath'] = np.mean(proj(
            np.c_[fy_bath, fz_bath], vec_pt_center)) * unit_scale
        fy_restore = data.particles['v_fry'].array
        fz_restore = data.particles['v_frz'].array
        df_force.loc[frame, 'restoring'] = np.mean(proj(
            np.c_[fy_restore, fz_restore], vec_pt_center)) * unit_scale
        data_bond = pipeline_bond.compute(frame)
        fyz_bond = data_bond.particles['Force'].array[:, [1, 2]]
        df_force.loc[frame, 'viscoelastic'] = np.mean(proj(
            fyz_bond, vec_pt_center)) * unit_scale
        fyz_tot = data.particles['Force'][:, [1, 2]]
        fy_FSI = fyz_tot[:, 0] - (fyz_bond[:, 0] + fy_bath + fy_restore + Fay)
        fz_FSI = fyz_tot[:, 1] - (fyz_bond[:, 1] + fz_bath + fz_restore + Faz + f_grav)
        df_force.loc[frame, 'FSI'] = np.mean(proj(
            np.c_[fy_FSI, fz_FSI], vec_pt_center)) * unit_scale

    writer = pd.ExcelWriter(f'{case_path}/force-ring{ringID}.xlsx')
    df_force.to_excel(writer, index=False)
    writer.close()


def extract_all_rings_strain():
    """
    Extract strains of rings from the dump file.
    """
    pipeline = import_file(f'{case_path}/0to1250000.dump', sort_particles=True)
    pipeline.modifiers.extend([
        mod.SelectTypeModifier(types={1, 3}),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])
    n_frames = pipeline.source.num_frames
    ring_strains = np.zeros((200, n_frames))
    for frame in trange(n_frames):
        data = pipeline.compute(frame)
        yz = data.particles['Position'][:, 1:]
        yz = yz.reshape(-1, n_yz, 2)
        center_yz = yz.mean(axis=1, keepdims=True)
        ring_strain = np.sqrt(((yz - center_yz) ** 2).sum(axis=-1)).mean(axis=1)
        ring_strains[:, frame] = (ring_strain - r_si) / r_si
    np.save(f'{case_path}/all_rings_strain', ring_strains)
    plt.imshow(ring_strains.T, aspect='auto')
    plt.show()


def interpolate_ring_strain(points):
    """
    Users must first run extract_all_rings_strain to get the data.
    :param points: ndarray with shape of (N,2), 0th col is time and 1st col is x
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    strain = np.load(f'{case_path}/all_rings_strain.npy')
    strain_finer = RGI((np.arange(strain.shape[1]) * 100,  # t (ms)
                        np.arange(strain.shape[0])),  # ringID
                       strain.T, method='linear')
    return strain_finer(points)


def plot_force_strain_ring(
        ringID, ax_force=None, ax_strain=None,
        names_force=None,
        # use strain of interface (averaged but high-resolution) or dump (accurate but low-resolution)
        which_strain='interface',
        positive_force='inward'
):
    """Used in combined_figures-contract_relax.py"""
    if ax_force is None and ax_strain is None:
        plt.rc('font', size=17, family='Arial')
        fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(10, 5))
        ax_force, ax_strain = ax
        external_ax = False
    else:
        external_ax = True
    df_force = pd.read_excel(f'{case_path}/force-ring{ringID}.xlsx')
    if which_strain == 'dump':
        df_force['strain'] = np.load(f'{case_path}/all_rings_strain.npy')[ringID]
    else:  # interface
        df_force['strain'] = np.r_[
            np.load(f'{case_path}/all_rings_strain.npy')[ringID, 0],
            np.load(f'{case_path}/interface/strain_tension_1250000.npz'
                    )['strain'].T[ringID, 99::100]
        ]
    df_force.rename(columns={each: each.capitalize() for each in df_force.columns if each != 'FSI'}, inplace=True)
    df_force['Time'] = np.arange(df_force.shape[0]) * 0.1
    styles = {
        'Gravity': 'C0-', 'Active': 'C1-',
        'Bath': 'C2-', 'FSI': 'C5-',
        'Restoring': 'C4-', 'Viscoelastic': 'C3-'
    }
    if names_force is None:
        names_force = styles.keys()
    if ax_force:
        if positive_force == 'outward':
            df_force[list(names_force)] *= -1
        df_force.plot(
            x='Time', y=names_force,
            ax=ax_force,
            style=styles, xlim=[0, 25],
            legend=False, lw=2,
            xlabel=''
        )
    if ax_strain:
        df_force.plot(
            x='Time', y='Strain',
            ax=ax_strain,
            style='k', xlim=[0, 25],
            legend=False, lw=2,
            xlabel=''
        )
    xticks = np.arange(0, 26, 5)
    if external_ax:
        return xticks, xticks
    else:
        ax_force.set_xticks(xticks)
        ax_force.set_xticklabels(['' for _ in xticks])
        ax_strain.set_xticks(xticks)
        ax_strain.set_xticklabels(xticks)
        ax_force.set_ylabel('Inward force (μN)', fontdict=font_label)
        ax_strain.set_xlabel('Time (s)', fontdict=font_label)
        ax_strain.set_ylabel('Strain', fontdict=font_label)
        plt.show()


def FSI_force_map(ax=None):
    from utils.mathfunc import proj
    from utils.id2x import x2ringID

    filename = f'{case_path}/FSI_force_map'
    if os.path.exists(filename + '.npy'):
        fi_FSI_all = np.load(filename + '.npy')
    else:
        pipeline = import_file(f'{case_path}/0to1250000.dump', sort_particles=True)
        pipeline.modifiers.extend([
            mod.SelectTypeModifier(types={1, 3}),
            mod.DeleteSelectedModifier(operate_on={'particles'})
        ])
        # Fbond.dump only have type 2, no need to delete type 1 and 3
        pipeline_bond = import_file(f'{case_path}/Fbond.dump', sort_particles=True)
        n_frames = pipeline.source.num_frames
        unit_scale = 1e6
        fi_FSI_all = np.zeros((n_frames, n_rings))
        for frame in trange(n_frames):
            data = pipeline.compute(frame)
            yz = data.particles['Position'].array[:, [1, 2]]
            yz = yz.reshape(n_rings, n_yz, 2)
            center_yz = yz.mean(axis=1, keepdims=True)
            vec_pt_center = center_yz - yz
            f_grav = -9.81 * m0
            Fay = data.particles['f_f_active[2]'].array
            Faz = data.particles['f_f_active[3]'].array
            fy_bath = data.particles['v_fy_bath'].array
            fz_bath = data.particles['v_fz_bath'].array
            fy_restore = data.particles['v_fry'].array
            fz_restore = data.particles['v_frz'].array
            data_bond = pipeline_bond.compute(frame)
            fyz_bond = data_bond.particles['Force'].array[:, [1, 2]]
            fyz_tot = data.particles['Force'][:, [1, 2]]
            fy_FSI = fyz_tot[:, 0] - (fyz_bond[:, 0] + fy_bath + fy_restore + Fay)
            fz_FSI = fyz_tot[:, 1] - (fyz_bond[:, 1] + fz_bath + fz_restore + Faz + f_grav)
            fi_FSI = proj(
                np.c_[fy_FSI, fz_FSI].reshape(n_rings, n_yz, 2), vec_pt_center)
            fi_FSI_all[frame] = np.mean(fi_FSI, axis=-1) * unit_scale
        np.save(filename, fi_FSI_all)
    if ax is None:
        plt.rc('font', **font_ticks)
        ax = plt.axes()
        external_ax = False
    else:
        external_ax = True
    ims = ax.imshow(fi_FSI_all, aspect='auto')
    cb = plt.colorbar(ims, location='right')
    cb.set_ticks(ticks=[-5, -4, -3, -2], labels=[-5, -4, -3, -2], fontdict=font_ticks)
    cb.set_label('FSI force (μN)', fontdict=font_label)
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels * 10
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax.set_ylim([yticks[-1], yticks[0]])
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)
    ax.set_ylabel('Time (s)', fontdict=font_label)
    ax.set_xlabel('x (mm)', fontdict=font_label)
    plt.show()


def ve_force_map(ax=None):
    """
    ve means viscoelastic
    """
    from utils.mathfunc import proj
    from utils.id2x import x2ringID

    filename = f'{case_path}/ve_force_map'
    if os.path.exists(filename + '.npy'):
        fi_ve_all = np.load(filename + '.npy')
    else:
        # Fbond.dump only have type 2, no need to delete type 1 and 3
        pipeline_bond = import_file(f'{case_path}/Fbond.dump', sort_particles=True)
        n_frames = pipeline_bond.source.num_frames
        unit_scale = 1e6
        fi_ve_all = np.zeros((n_frames, n_rings))
        for frame in trange(n_frames):
            data_bond = pipeline_bond.compute(frame)
            yz = data_bond.particles['Position'].array[:, [1, 2]]
            yz = yz.reshape(n_rings, n_yz, 2)
            center_yz = yz.mean(axis=1, keepdims=True)
            vec_pt_center = center_yz - yz
            fyz_ve = data_bond.particles['Force'].array[:, [1, 2]]
            fi_ve = proj(
                fyz_ve.reshape(n_rings, n_yz, 2), vec_pt_center)
            fi_ve_all[frame] = np.mean(fi_ve, axis=-1) * unit_scale
        np.save(filename, fi_ve_all)
    if ax is None:
        plt.rc('font', **font_ticks)
        ax = plt.axes()
        external_ax = False
    else:
        external_ax = True
    ims = ax.imshow(fi_ve_all, aspect='auto')
    cb = plt.colorbar(ims, location='right')
    # cb.set_ticks(ticks=[-5, -4, -3, -2], labels=[-5, -4, -3, -2], fontdict=font_ticks)
    # cb.set_label('ve force (μN)', fontdict=font_label)
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels * 10
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax.set_ylim([yticks[-1], yticks[0]])
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)
    ax.set_ylabel('Time (s)', fontdict=font_label)
    ax.set_xlabel('x (mm)', fontdict=font_label)
    plt.show()


def longitudinal_bond_force_map(ax=None):
    from utils.mathfunc import proj
    from utils.id2x import x2ringID

    filename = f'{case_path}/longitudinal_bond_force_map'
    if os.path.exists(filename + '.npy'):
        fi_ve_all = np.load(filename + '.npy')
    else:
        # Fbond.dump only have type 2, no need to delete type 1 and 3
        pipeline_bond = import_file(f'{case_path}/Fbond-longitudinal.dump', sort_particles=True)
        n_frames = pipeline_bond.source.num_frames
        unit_scale = 1e6
        fi_ve_all = np.zeros((n_frames, n_rings))
        for frame in trange(n_frames):
            data_bond = pipeline_bond.compute(frame)
            yz = data_bond.particles['Position'].array[:, [1, 2]]
            yz = yz.reshape(n_rings, n_yz, 2)
            center_yz = yz.mean(axis=1, keepdims=True)
            vec_pt_center = center_yz - yz
            fyz_ve = data_bond.particles['Force'].array[:, [1, 2]]
            fi_ve = proj(
                fyz_ve.reshape(n_rings, n_yz, 2), vec_pt_center)
            fi_ve_all[frame] = np.mean(fi_ve, axis=-1) * unit_scale
        np.save(filename, fi_ve_all)
    if ax is None:
        plt.rc('font', **font_ticks)
        ax = plt.axes()
        external_ax = False
    else:
        external_ax = True
    ims = ax.imshow(fi_ve_all, aspect='auto')
    cb = plt.colorbar(ims, location='right')
    # cb.set_ticks(ticks=[-5, -4, -3, -2], labels=[-5, -4, -3, -2], fontdict=font_ticks)
    # cb.set_label('ve force (μN)', fontdict=font_label)
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels * 10
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax.set_ylim([yticks[-1], yticks[0]])
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)
    ax.set_ylabel('Time (s)', fontdict=font_label)
    ax.set_xlabel('x (mm)', fontdict=font_label)
    plt.show()


def interpolate_FSI_force(points):
    """
    Users must first run FSI_force_map to get the data.
    :param points: ndarray with shape of (N,2), 0th col is time and 1st col is x
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    FSI_force = np.load(f'{case_path}/FSI_force_map.npy')
    FSI_force_finer = RGI((np.arange(FSI_force.shape[1]) * 100,  # t (ms)
                           np.arange(FSI_force.shape[0])),  # ringID
                          FSI_force, method='linear')
    return FSI_force_finer(points)


def interpolate_ve_force(points):
    """
    Users must first run ve_force_map to get the data.
    :param points: ndarray with shape of (N,2), 0th col is time and 1st col is x
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    ve_force = np.load(f'{case_path}/ve_force_map.npy')
    ve_force_finer = RGI((np.arange(ve_force.shape[1]) * 100,  # t (ms)
                          np.arange(ve_force.shape[0])),  # ringID
                         ve_force.T, method='linear')
    return ve_force_finer(points)


def plot_longitudinal_bond_force_multiframe(frames, ringIDmin, ringIDmax, ax=None):
    """
    Self-explanatory. The region is from ringIDmin to ringIDmax.
    :param frames: list
    :param ringIDmin: left bound of the region
    :param ringIDmax: right bound of the region
    :param ax: external Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    ve_force = np.load(f'{case_path}/longitudinal_bond_force_map.npy')
    # im = plt.imshow(ve_force[frames[0]:frames[-1], ringIDmin:ringIDmax],aspect='auto')
    # plt.colorbar(im)
    # plt.show()
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, len(frames)))
    for i, frame in enumerate(frames):
        ax.plot(ve_force[frame, ringIDmin:ringIDmax], color=colors[i], marker='v')
    if not external_ax:
        plt.show()


def chunk_vmag():
    """
    compute velocity magnitude of each sense chunk,
    in order to estimate the maximal allowed Ncallback
    """
    pipeline = import_file(f'{case_path}/to250000.dump', sort_particles=True)
    pipeline.modifiers.extend([
        mod.ExpressionSelectionModifier(
            expression=f'c_senseid<5 || c_senseid>96'),
        mod.DeleteSelectedModifier(operate_on={'particles'})
    ])
    n_sense_each = 2
    vmag_max = 0
    for frame in trange(pipeline.source.num_frames):
        data = pipeline.compute(frame)
        vmag = data.particles['Velocity Magnitude'].array
        vmag_max_cur = vmag.reshape(-1, n_yz * n_sense_each).mean(axis=1).max()
        if vmag_max_cur > vmag_max:
            vmag_max = vmag_max_cur
        print(vmag_max)
    delta_strain_max = 1e-3
    dt_lmp = 2e-5
    Ncallback_lmp_max = np.ceil(delta_strain_max * r_si / vmag_max / dt_lmp)
    print(Ncallback_lmp_max)


def extract_velocity(dump_name):
    from utils.sph_interpolate import extract_property
    from utils.id2x import SIP_ID2x

    n_sample = 184
    xs = SIP_ID2x(np.arange(n_sample) + 1)[:, np.newaxis] / 1000
    xyz = np.hstack((xs, np.zeros((n_sample, 2))))
    pipeline = import_file(f'{case_path}/{dump_name}', sort_particles=True)
    vel_sample = extract_property(
        pipeline,
        names_prop=['Velocity.X', 'Velocity.Y', 'Velocity.Z'],
        xyz=xyz,
        cutoff=h
    )
    np.savez(f'{case_path}/vel', vx=vel_sample[0], vy=vel_sample[1], vz=vel_sample[2])
    ims = plt.imshow(vel_sample[0].T, aspect='auto')
    plt.colorbar(ims, pad=0.07, shrink=0.7, anchor=(0.0, 0.9))
    plt.xlabel('Time')
    plt.ylabel('Intestine length')
    plt.show()


def draw_vmag_map():
    vel = np.load(f'{case_path}/vel.npz')
    v = np.concatenate((vel['vx'][:, :, np.newaxis],
                        vel['vy'][:, :, np.newaxis],
                        vel['vz'][:, :, np.newaxis]), axis=-1)
    vmag = np.sqrt(np.sum(v ** 2, axis=-1))
    ims = plt.imshow(vmag.T, aspect='auto')
    plt.colorbar(ims)
    plt.show()


def compare_strain_vx():
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    import plotly.express as px

    plt.rc('font', **font_ticks)
    fig, ax = plt.subplot_mosaic(
        [['kdex', '.'],
         ['scatter', 'kdey']],
        figsize=(10, 10),
        width_ratios=(2, 1), height_ratios=(1, 2),
        layout='constrained'
    )
    ax['kdex'].tick_params(axis="x", labelbottom=False)
    ax['kdey'].tick_params(axis="y", labelleft=False)
    color_line = ['#BA6518', '#2E6286', '#1F731F']
    # cmaps = ['Blues', 'Oranges', 'Greens']
    inset_kdex = ax['kdex'].inset_axes(
        [-35, 0.02, 31.5, 0.035],
        transform=ax['kdex'].transData,
        xlim=(-39, -5),
        ylim=(0, 0.012)
    )
    inset_kdey1 = ax['kdey'].inset_axes(
        [0.17, 5.5, 0.3, 5.5],
        transform=ax['kdey'].transData,
        xlim=(0, 0.03),
        ylim=(2.2, 7.6)
    )
    inset_kdey2 = ax['kdey'].inset_axes(
        [0.17, -15, 0.3, 11],
        transform=ax['kdey'].transData,
        xlim=(0, 0.06),
        ylim=(-14, -1.4)
    )
    for i, case_name in enumerate(cases_all):
        tid, cid = np.meshgrid(np.arange(250), np.arange(184) + 1)
        vx_sample = np.load(f'{RES_PATH}/{case_name}/vel_x.npy')
        vx_sample = vx_sample.T[:, 1:].flatten() * 1000
        strain = np.load(f'{RES_PATH}/{case_name}/ring_strain.npy')
        strain = strain.T[:, 1:].flatten() * 100

        ax['scatter'].scatter(strain, vx_sample, alpha=0.5, c=f'C{i}')
        xy = pd.DataFrame({'x': strain, 'y': vx_sample})
        sns.kdeplot(xy, x='x', ax=ax['kdex'], color=f'C{i}')
        sns.kdeplot(xy, x='x', ax=inset_kdex, color=f'C{i}')
        sns.kdeplot(xy, y='y', ax=ax['kdey'], color=f'C{i}')
        sns.kdeplot(xy, y='y', ax=inset_kdey1, color=f'C{i}')
        sns.kdeplot(xy, y='y', ax=inset_kdey2, color=f'C{i}')

        k, b = np.polyfit(strain, vx_sample, 1)
        vx_pred = k * strain + b
        ax['scatter'].plot(strain, vx_pred, color=color_line[i])
        print(f'Group {i}, k={k}, b={b}.')

        xy['cid'] = cid[:].flatten()
        xy['tid'] = tid[:].flatten()
        print(xy.shape[0])
        figure = px.scatter(xy, x='x', y='y', color='cid', hover_data=['cid', 'tid'])
        figure.show()

        # sns.kdeplot(xy, x='x', y='y', alpha=0.5, cmap=cmaps[i], fill=True, log_scale=True, ax=ax2)
    ax['scatter'].axhline(y=0, color='k', ls='--')
    ax['scatter'].axvline(x=0, color='k', ls='--')
    ax['kdex'].axvline(x=0, color='k', ls='--')
    ax['kdey'].axhline(y=0, color='k', ls='--')
    ax['kdex'].set_xlim(ax['scatter'].get_xlim())
    ax['kdey'].set_ylim(ax['scatter'].get_ylim())

    ax['scatter'].set_xlabel('Strain (%)', fontdict=font_label)
    ax['scatter'].set_ylabel('Axial velocity (mm/s)', fontdict=font_label)
    ax['kdex'].set_ylabel('Density (1/%)', fontdict=font_label)
    ax['kdey'].set_xlabel('Density (s/mm)', fontdict=font_label)
    inset_kdex.set_xlabel('')
    inset_kdex.set_ylabel('')
    inset_kdey1.set_ylabel('')
    inset_kdey2.set_ylabel('')
    mark_inset(ax['kdex'], inset_kdex, loc1=3, loc2=4, ec='k')
    inset_kdey1.set_xlabel('')
    inset_kdey1.set_ylabel('')
    mark_inset(ax['kdey'], inset_kdey1, loc1=3, loc2=2, ec='k')
    inset_kdey2.set_xlabel('')
    inset_kdey2.set_ylabel('')
    mark_inset(ax['kdey'], inset_kdey2, loc1=3, loc2=2, ec='k')

    # fig.savefig(f'{case_path}/viscosity_kde.png', dpi=300)
    plt.show()


def compare_Fa_strain():
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    import scipy.stats as stats
    import plotly.express as px
    from utils.mathfunc import detect_narrow_peaks_2d

    plt.rc('font', **font_ticks)
    fig, ax = plt.subplot_mosaic(
        [['kdex', '.'],
         ['scatter', 'kdey']],
        figsize=(10, 10),
        width_ratios=(2, 1), height_ratios=(1, 2),
        layout='constrained'
    )
    color_line = ['#BA6518', '#2E6286', '#1F731F']
    for i, case_name in enumerate(cases_all):
        tid, cid = np.meshgrid(np.arange(250), np.arange(184) + 1)
        st = np.load(f'{RES_PATH}/{case_name}/interface/strain_tension_1250000.npz')
        tension_ = st['tension'].T[:, 99::100] * 1e6
        _, valid = detect_narrow_peaks_2d(
            tension_, min_height=5, min_peak_prominence=3, rest_value=0.2, min_peak_interval=10)
        valid = ~valid
        valid[:5] = False
        valid[-5:] = False
        valid[:] = True
        valid = valid.flatten()
        tension_ = tension_.flatten()
        tension, tension_invalid = tension_[valid], tension_[~valid]

        strain_ = np.load(f'{RES_PATH}/{case_name}/ring_strain.npy')
        strain_ = strain_.T[:, 1:].flatten() * 100
        strain, strain_invalid = strain_[valid], strain_[~valid]
        ax['scatter'].scatter(tension, strain, alpha=0.3, c=f'C{i}')
        ax['scatter'].scatter(tension_invalid, strain_invalid, s=20, c='gray', alpha=0.3, zorder=0)
        xy = pd.DataFrame({'x': tension, 'y': strain})
        sns.kdeplot(xy, y='y', ax=ax['kdey'], color=f'C{i}')

        loc, scale = stats.expon.fit(tension, floc=0)
        lambda_hat = 1 / scale
        print(lambda_hat)
        tension_pred = np.linspace(0, tension.max(), 200)
        density_tension_pred = stats.expon.pdf(tension_pred, loc=loc, scale=scale)
        ax['kdex'].plot(tension_pred, density_tension_pred)
        k, b = np.polyfit(tension, strain, 1)
        strain_pred = k * tension + b
        ax['scatter'].plot(tension, strain_pred, color=color_line[i])
        print(f'Group {i}, k={k}, b={b}.')

        xy['cid'] = cid[:].flatten()[valid]
        xy['tid'] = tid[:].flatten()[valid]
        print(xy.shape[0])
        figure = px.scatter(xy, x='x', y='y', color='cid', hover_data=['cid', 'tid'])
        figure.show()
    ax['scatter'].set_ylim([-46.5, 21])
    ax['scatter'].set_xlabel('Active force (μN)', fontdict=font_label)
    ax['scatter'].set_ylabel('Strain (%)', fontdict=font_label)
    ax['kdex'].set_ylabel('Density (1/μN)', fontdict=font_label)
    ax['kdey'].set_xlabel('Density (1/%)', fontdict=font_label)
    plt.show()


def mixing_index(dump_name):
    from ovito.data import CutoffNeighborFinder

    cutoff = 0.0006
    probes = np.arange(0.004, 0.037, 0.004)[:, np.newaxis]
    n_probes = probes.shape[0]
    probes = np.hstack((probes, np.zeros((n_probes, 2))))
    pipeline = import_file(f'{case_path}/{dump_name}', sort_particles=True)
    pipeline.modifiers.extend([
        mod.SelectTypeModifier(types={1, 2}),
        mod.DeleteSelectedModifier(operate_on={'particles'}),
    ])
    data0 = pipeline.compute(0)
    finder = CutoffNeighborFinder(cutoff, data0)
    id_neighs_all = []
    for probe in probes:
        neighs = finder.find_at(probe)
        id_neighs_all.append([neigh.index for neigh in neighs])
    n_frames = pipeline.source.num_frames
    dists_mean = np.zeros((n_probes, n_frames))
    for frame in trange(n_frames):
        data = pipeline.compute(frame)
        for i_probe in range(n_probes):
            pos_neigh = data.particles['Position'][id_neighs_all[i_probe]]
            pos_center = pos_neigh.mean(axis=0, keepdims=True)
            dists_mean[i_probe, frame] = np.sqrt(((pos_neigh - pos_center) ** 2).sum(axis=1)).mean()
    np.save(f'{case_path}/dists_mean', dists_mean)
    plt.plot(np.arange(pipeline.source.num_frames), dists_mean.T)
    plt.show()


def compare_mixing_index():
    x = np.arange(251)
    for i, case_name in enumerate(cases_all):
        dists_mean = np.load(f'{RES_PATH}/{case_name}/dists_mean.npy')
        dists_mean_avg = dists_mean.mean(axis=0)
        dists_mean_std = dists_mean.std(axis=0)
        plt.plot(x, dists_mean_avg)
        plt.fill_between(
            x, y1=dists_mean_avg - 1.96 * dists_mean_std, y2=dists_mean_avg + 1.96 * dists_mean_std,
            alpha=0.2, ec='k')
    plt.show()


if __name__ == '__main__':
    # flow_rate('in_out_flow_0to1250000.yml')
    # flow_rate_paraview()
    # flow_rate_map()
    # outflow_compare()
    # outflow_compare_paraview()
    # intraluminal_pressure('0to1250000.dump')
    # extract_force_strain_atom()
    # ===========================
    # from tqdm import tqdm
    # for frame in tqdm([80, 88, 217]):
    #     draw_atom_force_arrow(frame, if_annotate=False, spine=False, transparent_bg=True)
    # draw_atom_force_arrow(80, transparent_bg=True)
    # ===========================
    force_time_atom_paper()
    # plot_atom_force_strain(130)
    # combine_atom_force_strain_arrow()
    # force_time_atom()
    # extract_force_ring(ringID=8)
    # fig, ax = plt.subplots(2,1,sharex=True)
    # plot_force_strain_ring(ringID=8)
    # ax[1].set_xlabel('Time (s)', fontdict=font_label)
    # ax[1].set_ylabel('Strain', fontdict=font_label)
    # ax[0].set_ylabel('Inward force (μN)', fontdict=font_label)
    # plt.show()
    # FSI_force_map()
    # ve_force_map()
    # longitudinal_bond_force_map()
    # plot_longitudinal_bond_force_multiframe(list(range(174, 181, 1)), ringIDmin=25, ringIDmax=37)
    # print(interpolate_FSI_force(np.array([[1, 2], [2, 3]])))
    # chunk_vmag()
    # strain_pressure()
    # extract_velocity('0to1250000.dump')
    # draw_vmag_map()
    # compare_strain_vx()
    # compare_Fa_strain()
    # extract_all_rings_strain()
    # interpolate_ring_strain(np.array([[1, 2]]))
    # mixing_index('merge0to1250000.dump')
    # compare_mixing_index()
