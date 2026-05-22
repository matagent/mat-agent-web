"""
MatAgent MCP Server - 材料科学计算工具集
提供材料查询、结构建模、VASP任务管理等功能
"""

# ============ 标准库导入 ============
import os
import io
import json
import atexit
import signal
import shutil
import tempfile
import multiprocessing
from datetime import datetime
from warnings import simplefilter
from itertools import product
from typing import Optional, Dict, Any, List

from pydantic_core import Url

# 忽略 FutureWarning
simplefilter(action='ignore', category=FutureWarning)

# ============ 第三方库导入 ============
import numpy as np
import pandas as pd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage
from pydantic import BaseModel

from mp_api.client import MPRester
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import Vasprun as VasprunOutput
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer, BV_PARAMS

from ase.io import write
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import bulk

# ============ 本地模块导入 ============
import loadenv
import databasemanage
import tryssh
import oqmd
import aflow_search
import flask_server

# ============ 全局配置 ============
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["font.family"] = ["serif"]
plt.rcParams['axes.unicode_minus'] = False

# MCP 实例
mcp = FastMCP(name="MatAgent")

# 全局子进程列表
child_processes: list[tuple[multiprocessing.Process, str]] = []

# ============ 环境配置加载 ============
config = loadenv.Config()
if not config.validate_config():
    raise EnvironmentError("请设置必要的环境变量")

MY_API_KEY = config.get_api_key()
IP = config.get_ip()
HOST = config.get_host()
PORT = config.get_port()
USERNAME = config.get_username()
PASSWORD = config.get_password()

# ============ 进程清理函数 ============
def cleanup_child_processes():
    """在主进程退出时尝试优雅终止所有子进程并删除临时文件目录"""
    for p, temp_dir in list(child_processes):
        try:
            if p.is_alive():
                p.terminate()
                p.join(3)
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(1)
        except Exception:
            pass
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            child_processes.remove((p, temp_dir))
        except ValueError:
            pass

def _handle_exit(signum, frame):
    cleanup_child_processes()
    os._exit(0)

atexit.register(cleanup_child_processes)
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


# ============ 工具函数 ============
def get_plot_url(img_buffer: io.BytesIO) -> str:
    """获取图片的 URL"""
    return matfileserver.add_image(img_buffer)


def _create_error_image(error_message: str) -> io.BytesIO:
    """创建错误信息图片"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, f"❌ {error_message}",
            ha='center', va='center', fontsize=12, color='red')
    ax.set_axis_off()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def apply_scientific_style():
    """优化后的出版级绘图风格"""
    okabe_ito = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                 '#0072B2', '#D55E00', '#CC79A7']
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler(color=okabe_ito),
        'figure.dpi': 150
    })
    return okabe_ito


def _get_density_array(dos_obj):
    """从 Dos 对象中提取密度数组"""
    assert hasattr(dos_obj, "densities") and dos_obj.densities, "Dos 对象不包含密度数据"
    return list(dos_obj.densities.values())[0]


def _smooth_dos(y, window=7, order=3):
    """Savitzky-Golay 平滑 DOS 曲线，保留峰形特征"""
    try:
        from scipy.signal import savgol_filter
        if len(y) <= window:
            return y
        return savgol_filter(y, window_length=min(window, len(y) - (1 - len(y) % 2)), polyorder=min(order, window - 1))
    except ImportError:
        # 回退：简单移动平均
        window = min(window, len(y))
        if window <= 2:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode='same')


def _enhance_for_plot(atoms: Atoms, tolerance: float = 0.05) -> Atoms:
    """专门为可视化增强 Atoms：将边界原子复制到相对的边界、棱和顶点"""
    cell = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    
    new_scaled = []
    new_symbols = []
    
    offsets = list(product([0, 1], repeat=3))
    
    for pos, symbol in zip(scaled_positions, symbols):
        near_zero = np.isclose(pos, 0, atol=tolerance)
        
        for off in offsets:
            if any(o == 1 and not nz for o, nz in zip(off, near_zero)):
                continue
            new_scaled.append(pos + off)
            new_symbols.append(symbol)
            
    enhanced = Atoms(symbols=new_symbols, 
                    scaled_positions=new_scaled, 
                    cell=cell, 
                    pbc=True)
    return enhanced


# ============ 结构可视化 ============
def visualize_structure(structure: Structure) -> str:
    """可视化晶体结构的3D交互式网页"""
    formula = structure.composition.reduced_formula
    atoms = AseAtomsAdaptor.get_atoms(structure)

    temp_dir = tempfile.mkdtemp(prefix=f"{formula}_custom_")
    html_path = os.path.join(temp_dir, f"{formula}_custom_3d.html")
    write(html_path, atoms, format='html')

    url = matfileserver.add_html_with_info(structure, html_path)
    return url


def get_structure_plot(structure: Structure,
                       repeat: bool = True, 
                       rotation: str = '10x,10y,0z') -> dict:
    """输入指定的晶体结构并返回预览图"""
    try:
        atoms = structure.to_ase_atoms()
        atoms.wrap()
        
        enhanced_atoms = _enhance_for_plot(atoms=atoms)
        
        fig, ax = plt.subplots(figsize=(16, 16))
        
        plot_atoms(
            enhanced_atoms,
            ax,
            rotation=rotation,
            show_unit_cell=2,
        )
        
        analyzer = SpacegroupAnalyzer(structure)
        spacegroup = analyzer.get_space_group_symbol()
        
        a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
        alpha, beta, gamma = structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma
        formula = structure.composition.formula
        
        info_text = (
            f"Formula: {formula}\n"
            f"Space group: {spacegroup}\n"
            f"Lattice parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å\n"
            f"Angles: α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°\n"
            f"Atoms in unit cell: {len(structure)}\n"
            f"Total atoms shown: {len(atoms)}"
        )
        
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_axis_off()
        ax.set_title(f"Crystal Structure Visualization of {structure.composition.reduced_formula}", 
                    fontsize=14, fontweight='bold')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return {"Image": get_plot_url(img_buffer), "error": None}
        
    except Exception as e:
        return {"Image": get_plot_url(_create_error_image(f"构建失败: {str(e)}")), "error": e}


# ============ VASP 绘图函数 ============
def plot_vasp_band(xml_path, kpoints_path):
    """使用 Pymatgen 绘制高质量能带图"""
    try:
        run = Vasprun(xml_path, parse_projected_eigen=False)
        bs = run.get_band_structure(kpoints_filename=kpoints_path, line_mode=True)

        is_metal = bs.is_metal()
        gap_info = bs.get_band_gap()
        
        results = {
            "is_metal": is_metal,
            "gap": gap_info['energy'],
            "fermi_energy": bs.efermi,
        }

        plotter = BSPlotter(bs)
        plt_module = plotter.get_plot()
        fig = plt.gcf() 
        ax = plt.gca()
        
        xticks = ax.get_xticks()
        labels = [label.get_text() for label in ax.get_xticklabels()]
        fixed_labels = [l.replace('GAMMA', r'$\Gamma$') for l in labels]
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(fixed_labels, fontsize=20)
        ax.set_ylabel(r'$E - E_f$ (eV)', fontsize=20)
        ax.set_title('Band Structure', fontsize=22, pad=20)
        ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2, zorder=1)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
            
        return {"Image": get_plot_url(buf), "data": results, "error": None}

    except Exception as e:
        return {"Image": None, "error": str(e)}


# ============ 共享工具函数 ============

def _clean_numpy(obj):
    """递归将 numpy 类型转换为 Python 原生类型，确保 JSON 可序列化。"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _clean_numpy(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# ============ 计算服务器连接检查 ============

def _check_connection() -> dict | None:
    """检查计算服务器连接状态，未连接时返回错误信息字典"""
    if connection is None:
        return {"error": "计算服务器未连接", "message": "SSH 连接失败，VASP/计算相关工具当前不可用。材料搜索、结构查询、带隙预测等其他工具不受影响"}


# ============ 共享结构分析函数 ============

def analyze_structure(structure: Structure, detail: str = "normal") -> dict:
    """对晶体结构进行综合分析：Wyckoff位置、键长、键角、配位环境。

    所有 get_material_structure_* 和 build_structure 工具共用此函数。

    Args:
        structure: pymatgen Structure 对象
        detail: "brief" (仅统计摘要), "normal" (统计+精选列表, 默认), "full" (完整逐条数据)
    """
    from collections import defaultdict

    analysis: dict = {"detail": detail}

    # --- Wyckoff 位置分析 ---
    try:
        sga = SpacegroupAnalyzer(structure)
        sym_dataset = sga.get_symmetry_dataset()
        wyckoff_labels = list(sym_dataset.get("wyckoffs", []))
        equivalent_atoms = [int(e) for e in sym_dataset.get("equivalent_atoms", [])]
        sites = structure.sites

        wyckoff_list = []
        seen_indices = set()
        for i, site in enumerate(sites):
            if i in seen_indices:
                continue
            equiv_indices = [j for j, eq in enumerate(equivalent_atoms) if eq == equivalent_atoms[i]]
            seen_indices.update(equiv_indices)

            wyckoff_list.append({
                "element": str(site.specie.element) if hasattr(site.specie, 'element') else site.species_string,
                "wyckoff_letter": wyckoff_labels[i] if i < len(wyckoff_labels) else "?",
                "multiplicity": len(equiv_indices),
                "site_symmetry": sym_dataset.get("site_symmetry_symbols", [""])[i] if i < len(
                    sym_dataset.get("site_symmetry_symbols", [])
                ) else "",
                "fractional_coordinates": [round(c, 4) for c in site.frac_coords[:3]],
            })

        analysis["wyckoff_positions"] = wyckoff_list
        analysis["space_group_symbol"] = sga.get_space_group_symbol()
        analysis["space_group_number"] = int(sga.get_space_group_number())
        analysis["crystal_system"] = sga.get_crystal_system()
    except Exception as e:
        analysis["wyckoff_error"] = str(e)

    # --- 键长和配位分析 ---
    try:
        try:
            structure.add_oxidation_state_by_guess()
        except Exception:
            pass

        cnn = CrystalNN()
        bond_list = []      # 所有键（含重复方向）
        all_nn_info = {}
        for i in range(len(structure)):
            try:
                nn_info = cnn.get_nn_info(structure, i)
            except Exception:
                nn_info = []
            all_nn_info[i] = []
            for neighbor in nn_info:
                j = neighbor["site_index"]
                dist = round(float(structure.get_distance(i, j)), 4)
                bond_list.append({
                    "pair": f"{str(structure[i].specie.element)}-{str(structure[j].specie.element)}",
                    "distance": dist,
                })
                image = neighbor.get("image", [0, 0, 0])
                neigh_cart = structure[j].coords + np.dot(image, structure.lattice.matrix)
                all_nn_info[i].append({
                    "site_index": j,
                    "cart_coords": neigh_cart,
                    "site": structure[j],
                })

        # 按元素对去重
        seen_keys = set()
        deduped = []
        for b in bond_list:
            k = b["pair"]
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append(b)
        analysis["bond_count"] = len(deduped)

        # 键长统计摘要（按元素对分组）
        pair_stats = defaultdict(list)
        for b in bond_list:
            pair_stats[b["pair"]].append(b["distance"])
        bond_summary = []
        for pair, dists in sorted(pair_stats.items()):
            bond_summary.append({
                "pair": pair,
                "count": len(dists),
                "min": round(min(dists), 4),
                "max": round(max(dists), 4),
                "mean": round(sum(dists) / len(dists), 4),
            })
        analysis["bond_summary"] = sorted(bond_summary, key=lambda x: x["mean"])

        # 完整键列表：仅 full 模式输出
        if detail == "full":
            analysis["bonds"] = [{"pair": b["pair"], "distance": b["distance"]} for b in bond_list]

        # 配位数：先计算每个位点的 CN
        site_cns = []
        for i in range(len(structure)):
            try:
                cn_val = float(cnn.get_cn(structure, i))
            except Exception:
                cn_val = 0.0
            site_cns.append(cn_val)

        # 按元素统计摘要
        cn_by_elem = defaultdict(list)
        for i in range(len(structure)):
            elem = str(structure[i].specie.element)
            cn_by_elem[elem].append(site_cns[i])
        cn_summary = {}
        for elem, cns in sorted(cn_by_elem.items()):
            cn_summary[elem] = {
                "min": round(min(cns), 2), "max": round(max(cns), 2),
                "mean": round(sum(cns) / len(cns), 2), "count": len(cns),
            }
        analysis["coordination_summary"] = cn_summary

        # 逐位点配位数：brief 模式不输出
        if detail != "brief":
            analysis["coordination"] = {
                f"{str(structure[i].specie.element)}_{i}": {
                    "element": str(structure[i].specie.element),
                    "site_index": i,
                    "coordination_number": round(float(site_cns[i]), 2),
                }
                for i in range(len(structure))
            }

    except Exception as e:
        analysis["bond_error"] = str(e)

    # --- 键角分析 ---
    try:
        angle_list = []
        for i in range(len(structure)):
            neighbors = all_nn_info.get(i, [])
            if len(neighbors) < 2:
                continue
            vertex_coord = structure[i].coords
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    na = neighbors[a_idx]
                    nb = neighbors[b_idx]
                    v1 = np.asarray(na["cart_coords"]) - np.asarray(vertex_coord)
                    v2 = np.asarray(nb["cart_coords"]) - np.asarray(vertex_coord)
                    norm1 = float(np.linalg.norm(v1))
                    norm2 = float(np.linalg.norm(v2))
                    if norm1 < 0.01 or norm2 < 0.01:
                        continue
                    cos_angle = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
                    angle_deg = round(float(np.degrees(np.arccos(cos_angle))), 2)
                    if 10 < angle_deg < 179:
                        triplet = f"{str(na['site'].specie.element)}-{str(structure[i].specie.element)}-{str(nb['site'].specie.element)}"
                        angle_list.append({"triplet": triplet, "angle_deg": angle_deg})

        analysis["angle_count"] = len(angle_list)

        # 键角统计摘要（按三元组分组）
        triplet_stats = defaultdict(list)
        for a in angle_list:
            triplet_stats[a["triplet"]].append(a["angle_deg"])
        angle_summary = []
        for triplet, angles in sorted(triplet_stats.items()):
            angle_summary.append({
                "triplet": triplet,
                "count": len(angles),
                "min": min(angles),
                "max": max(angles),
                "mean": round(sum(angles) / len(angles), 2),
            })
        analysis["angle_summary"] = angle_summary

        # 完整角度列表：仅 full 模式输出（每组 top 8）
        if detail == "full":
            grouped = defaultdict(list)
            for a in angle_list:
                grouped[a["triplet"]].append(a)
            flat_angles = []
            for triplet in sorted(grouped.keys()):
                items = grouped[triplet]
                items.sort(key=lambda x: min(abs(x["angle_deg"] - 180), abs(x["angle_deg"] - 109.5)))
                flat_angles.extend(items[:8])
            analysis["angles"] = flat_angles

    except Exception as e:
        analysis["angle_error"] = str(e)

    # --- 键价和 (BVS) 分析 ---
    try:
        bva = BVAnalyzer()
        estimated_valences = bva.get_valences(structure)

        # 按元素统计
        from collections import defaultdict as _dd
        bvs_by_elem = _dd(list)
        for i, site in enumerate(structure):
            elem = str(site.specie.element)
            bvs_by_elem[elem].append(estimated_valences[i])

        bvs_summary = []
        for elem, vals in sorted(bvs_by_elem.items()):
            bvs_summary.append({
                "element": elem,
                "bv_sum": round(sum(vals) / len(vals), 2) if len(vals) == 1 else [round(v, 2) for v in vals],
                "expected_oxi": round(abs(sum(vals) / len(vals)), 2) if len(vals) == 1 else None,
            })
        analysis["bvs"] = bvs_summary

        # 逐位点 BVS (normal/full 模式)
        if detail != "brief":
            bvs_sites = []
            for i, site in enumerate(structure):
                bvs_sites.append({
                    "site_index": i,
                    "element": str(site.specie.element),
                    "bvs_valence": round(estimated_valences[i], 2) if isinstance(estimated_valences[i], float) else estimated_valences[i],
                })
            analysis["bvs_sites"] = bvs_sites

        # full 模式：计算每个键的键价贡献
        if detail == "full" and "bonds" in analysis:
            for b in analysis["bonds"]:
                r = b["distance"]
                pair = b["pair"].split("-")
                if len(pair) == 2:
                    cation_elem = pair[0]
                    anion_elem = pair[1]
                    cation_params = BV_PARAMS.get(cation_elem)
                    anion_params = BV_PARAMS.get(anion_elem)
                    if cation_params:
                        R0 = cation_params["r"]
                        b_const = cation_params.get("c", 0.37)
                        b["bv_contribution"] = round(float(np.exp((R0 - r) / b_const)), 4)
                    elif anion_params:
                        R0 = anion_params["r"]
                        b_const = anion_params.get("c", 0.37)
                        b["bv_contribution"] = round(float(np.exp((R0 - r) / b_const)), 4)

    except Exception as e:
        analysis["bvs_error"] = str(e)

    return _clean_numpy(analysis)


def plot_vasp_dos_analysis(vasprun_path="vasprun.xml", material_name="Material", smooth: int = 0):
    """主接口：解析 VASP 数据并生成 2x3 综合分析图

    Args:
        vasprun_path: vasprun.xml 路径
        material_name: 材料名称
        smooth: Savitzky-Golay 平滑窗口大小，0 或 1 则不平滑 (默认 0)
    """
    try:
        print(f"正在解析 {vasprun_path}...")
        vr = VasprunOutput(vasprun_path, parse_dos=True)
        complete_dos = vr.complete_dos

        assert complete_dos is not None, "无法从 vasprun 提取 CompleteDos"
        assert hasattr(complete_dos, "energies"), "CompleteDos 对象缺失能量数据"

        energies = complete_dos.energies - complete_dos.efermi
        tdos_raw = _get_density_array(complete_dos)
        element_dos = complete_dos.get_element_dos()

        # DOS数据分析（用原始数据）
        dos_analysis = _analyze_dos_data(energies, tdos_raw, element_dos)

        # 平滑处理（仅用于绘图）
        if smooth and smooth > 1:
            tdos_plot = _smooth_dos(tdos_raw, window=smooth)
            element_dos_plot = {
                el: _smooth_dos(_get_density_array(d), window=smooth)
                for el, d in (element_dos or {}).items()
            }
        else:
            tdos_plot = tdos_raw
            element_dos_plot = None

        # 绘图逻辑 - 2行3列布局
        colors = apply_scientific_style()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Electronic Structure Analysis: {material_name}', fontweight='bold')

        # (A) Total DOS
        ax = axes[0, 0]
        ax.plot(energies, tdos_plot, color='black', lw=1.5, label='Total DOS')
        ax.fill_between(energies, 0, tdos_plot, where=(energies < 0), color='gray', alpha=0.2)
        ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1, label='$E_F$')
        
        if 'band_gap' in dos_analysis:
            gap_text = f"Band gap: {dos_analysis['band_gap']:.3f} eV"
            ax.text(0.05, 0.95, gap_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('(A) Total Density of States')
        ax.set_ylabel('DOS (states/eV)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

        # (B) Element Projected DOS
        ax = axes[0, 1]
        if element_dos:
            for i, (el, dos_obj) in enumerate(element_dos.items()):
                dens = (element_dos_plot.get(el) if element_dos_plot
                        else _get_density_array(dos_obj))
                ax.plot(energies, dens, label=str(el), lw=1.3)
            ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1)
            ax.set_title('(B) Element Projected DOS')
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No Element PDOS found", ha='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # (C) Near-Fermi Region (Zoomed)
        ax = axes[0, 2]
        mask = (energies > -4) & (energies < 4)
        ax.plot(energies[mask], tdos_plot[mask], color='black', lw=1.2)
        ax.fill_between(energies[mask], 0, tdos_plot[mask], where=(energies[mask] < 0), color='#56B4E9', alpha=0.3)
        ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1)
        
        if 'dos_at_ef_exact' in dos_analysis:
            fermi_dos_text = f"DOS(E$_F$) = {dos_analysis['dos_at_ef_exact']:.3f}"
            ax.text(0.05, 0.95, fermi_dos_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_title('(C) Near-Fermi Region (±4 eV)')
        ax.set_xlabel('Energy - $E_F$ (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.grid(True, alpha=0.3)

        # (D) Integrated DOS
        ax = axes[1, 0]
        if len(energies) > 1:
            de = energies[1] - energies[0]
            integrated = np.cumsum(tdos_plot) * de
            ax.plot(energies, integrated, color='#009E73', lw=1.5)
            
            if 'total_integrated_dos' in dos_analysis:
                total_electrons = dos_analysis['total_integrated_dos']
                ax.text(0.05, 0.95, f"Total e$^-$: {total_electrons:.1f}", 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "Insufficient data\nfor integration", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('(D) Integrated DOS')
        ax.set_ylabel('Cumulative Electrons')
        ax.set_xlabel('Energy - $E_F$ (eV)')
        ax.grid(True, alpha=0.3)

        # (E) 元素贡献饼图
        ax = axes[1, 1]
        if element_dos and 'element_contributions' in dos_analysis:
            element_contributions = dos_analysis['element_contributions']
            
            elements = []
            fermi_contributions = []
            
            for el, contrib in element_contributions.items():
                elements.append(el)
                fermi_contributions.append(contrib['fermi_contribution'])
            
            valid_indices = [i for i, val in enumerate(fermi_contributions) if val > 0]
            if valid_indices and len(valid_indices) > 1:
                elements = [elements[i] for i in valid_indices]
                fermi_contributions = [fermi_contributions[i] for i in valid_indices]
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(elements)))
                
                wedges, texts, autotexts = ax.pie(
                    fermi_contributions, 
                    labels=elements, 
                    colors=colors_pie,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 9}
                )
                
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontsize(8)
                    autotext.set_fontweight('bold')
                
                ax.set_title('(E) Element Contribution at Fermi Level')
            else:
                ax.text(0.5, 0.5, "Insufficient element\ncontributions data", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, "No element contribution data", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        # (F) DOS峰位分析图
        ax = axes[1, 2]
        ax.plot(energies, tdos_plot, color='black', lw=1.2, alpha=0.7, label='Total DOS')
        
        if 'major_peaks' in dos_analysis and dos_analysis['major_peaks']:
            peaks = dos_analysis['major_peaks']
            peak_energies = [p['energy'] for p in peaks]
            peak_heights = [p['dos_height'] for p in peaks]
            
            peak_colors = plt.cm.viridis(np.linspace(0, 1, len(peaks)))
            for i, (energy, height, color) in enumerate(zip(peak_energies, peak_heights, peak_colors)):
                ax.scatter(energy, height, color=color, s=80, zorder=5, 
                          edgecolors='black', linewidth=1)
                label_text = f"P{i+1}: {energy:.2f} eV"
                ax.annotate(label_text, 
                           xy=(energy, height),
                           xytext=(energy, height * 1.1),
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
            
            peak_table_data = []
            for i, peak in enumerate(peaks[:3]):
                peak_table_data.append([
                    f"P{i+1}",
                    f"{peak['energy']:.2f} eV",
                    f"{peak['dos_height']:.2f}"
                ])
            
            if peak_table_data:
                table = ax.table(cellText=peak_table_data,
                                colLabels=['Peak', 'Energy', 'DOS'],
                                cellLoc='center',
                                loc='upper right',
                                bbox=[0.65, 0.6, 0.3, 0.3])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
        else:
            ax.plot(energies, tdos_plot, color='black', lw=1.5)
            ax.text(0.5, 0.5, "No peak analysis available", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1, label='$E_F$')
        ax.set_title('(F) DOS Peak Analysis')
        ax.set_xlabel('Energy - $E_F$ (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        info_dict = {
            "material": material_name,
            "efermi": float(vr.complete_dos.efermi),
            "formula": vr.final_structure.composition.reduced_formula,
            "dos_analysis": dos_analysis
        }
        
        if 'band_gap' in dos_analysis:
            info_dict["band_gap_summary"] = {
                "value": dos_analysis['band_gap'],
                "type": dos_analysis.get('gap_type', 'unknown'),
                "vbm": dos_analysis.get('valence_band_max', None),
                "cbm": dos_analysis.get('conduction_band_min', None)
            }
        
        if 'major_peaks' in dos_analysis:
            info_dict["peak_summary"] = {
                "num_peaks": len(dos_analysis['major_peaks']),
                "main_peaks": dos_analysis['major_peaks'][:3] if len(dos_analysis['major_peaks']) >= 3 else dos_analysis['major_peaks']
            }
        
        if 'element_contributions' in dos_analysis:
            info_dict["element_contribution_summary"] = dos_analysis['element_contributions']

        return {
            "info": info_dict,
            "Image": get_plot_url(buf)
        }

    except AssertionError as ae:
        print(f"数据检查未通过: {ae}")
        return {"error": str(ae)}
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def _analyze_dos_data(energies, tdos_plot, element_dos):
    """分析DOS数据，返回带隙、费米能级处DOS等关键信息"""
    analysis_results = {}
    
    if len(energies) > 1:
        de = energies[1] - energies[0]
        analysis_results['energy_step'] = de
    
    valence_mask = energies < 0
    conduction_mask = energies > 0
    
    if np.any(valence_mask) and np.any(conduction_mask):
        valence_energies = energies[valence_mask]
        valence_dos = tdos_plot[valence_mask]
        valence_nonzero = valence_dos > 1e-6
        if np.any(valence_nonzero):
            vbm_index = np.argmax(valence_energies[valence_nonzero])
            vbm_energy = valence_energies[valence_nonzero][vbm_index]
            vbm_dos = valence_dos[valence_nonzero][vbm_index]
            analysis_results['valence_band_max'] = float(vbm_energy)
            analysis_results['vbm_dos'] = float(vbm_dos)
        
        conduction_energies = energies[conduction_mask]
        conduction_dos = tdos_plot[conduction_mask]
        conduction_nonzero = conduction_dos > 1e-6
        if np.any(conduction_nonzero):
            cbm_index = np.argmin(conduction_energies[conduction_nonzero])
            cbm_energy = conduction_energies[conduction_nonzero][cbm_index]
            cbm_dos = conduction_dos[conduction_nonzero][cbm_index]
            analysis_results['conduction_band_min'] = float(cbm_energy)
            analysis_results['cbm_dos'] = float(cbm_dos)
            
            if 'valence_band_max' in analysis_results:
                band_gap = float(cbm_energy - vbm_energy)
                analysis_results['band_gap'] = band_gap
                analysis_results['gap_type'] = 'direct' if abs(band_gap - (cbm_energy - vbm_energy)) < 0.01 else 'indirect'
    
    fermi_window = 0.05
    fermi_mask = (energies > -fermi_window) & (energies < fermi_window)
    if np.any(fermi_mask):
        fermi_dos_values = tdos_plot[fermi_mask]
        analysis_results['dos_at_fermi'] = float(np.mean(fermi_dos_values))
        analysis_results['fermi_window_avg'] = float(np.mean(fermi_dos_values))
        if len(energies) > 1:
            dos_at_ef = float(np.interp(0, energies, tdos_plot))
            analysis_results['dos_at_ef_exact'] = dos_at_ef
    
    if len(energies) > 1 and 'energy_step' in analysis_results:
        de = analysis_results['energy_step']
        total_electrons = float(np.sum(tdos_plot) * de)
        analysis_results['total_integrated_dos'] = total_electrons
    
    if np.any(valence_mask) and 'energy_step' in analysis_results:
        de = analysis_results['energy_step']
        valence_integral = float(np.sum(tdos_plot[valence_mask]) * de)
        analysis_results['valence_integrated_dos'] = valence_integral
    
    if np.any(conduction_mask) and 'energy_step' in analysis_results:
        de = analysis_results['energy_step']
        conduction_integral = float(np.sum(tdos_plot[conduction_mask]) * de)
        analysis_results['conduction_integrated_dos'] = conduction_integral
    
    element_contributions = {}
    if element_dos:
        for el, dos_obj in element_dos.items():
            el_dens = _get_density_array(dos_obj)
            if np.any(fermi_mask):
                el_fermi_contrib = float(np.mean(el_dens[fermi_mask]))
                if 'energy_step' in analysis_results:
                    de = analysis_results['energy_step']
                    element_contributions[str(el)] = {
                        'fermi_contribution': el_fermi_contrib,
                        'total_contribution': float(np.sum(el_dens) * de)
                    }
                else:
                    element_contributions[str(el)] = {
                        'fermi_contribution': el_fermi_contrib,
                        'total_contribution': float(np.sum(el_dens))
                    }
        analysis_results['element_contributions'] = element_contributions
    
    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(tdos_plot, height=0.1, distance=10)
        if len(peaks) > 0:
            peak_info = []
            for i, peak_idx in enumerate(peaks[:5]):
                peak_info.append({
                    'energy': float(energies[peak_idx]),
                    'dos_height': float(tdos_plot[peak_idx]),
                    'relative_to_fermi': float(energies[peak_idx])
                })
            analysis_results['major_peaks'] = peak_info
    except ImportError:
        print("scipy未安装，跳过峰位分析")
    except Exception as e:
        print(f"峰位分析失败: {e}")
    
    return analysis_results


# ============ VASP 结果提取函数 ============
def extract_relax_info(task_directory: str, get_plot: bool = True, visualize: bool = False) -> dict:
    """提取结构优化任务的结果信息"""
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.extract_relax_info(task_directory)
                if result:
                    break
            
            # 从 CONTCAR 文件重新读取 Structure 对象用于可视化
            structure = None
            if result and "local_files" in result and "contcar" in result["local_files"]:
                try:
                    contcar_path = result["local_files"]["contcar"]
                    structure = Structure.from_file(contcar_path)
                except Exception as e:
                    print(f"读取 CONTCAR 失败: {e}")
            
            if visualize and structure is not None:
                structure_url = visualize_structure(structure)
                result["3d_image_url"] = structure_url
            if get_plot and structure is not None:
                res = get_structure_plot(structure)
                image = res["Image"]
                result["error"] = res['error']
                result["image_url"] = image
            result.pop("structure", None)  
            return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_scf_info(task_directory: str) -> dict:
    """提取自洽计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_scf_info(task_directory)
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_band_info(task_directory: str, plot_band: bool = True) -> dict:
    """提取能带计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_band_info(task_directory)
            if plot_band:
                res = plot_vasp_band(xml_path=result['local_files']['vasprun.xml'],
                                    kpoints_path=result['local_files']['KPOINTS'])
                if not res["error"]:
                    image = res["Image"]
                    res.pop("Image")
                    result.update({"image_url": image, "plot_info": res, "message": "绘图成功"})
                else:
                    res.pop("Image")
                    result.update({"plot_info": res, "message": "绘图失败"})
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_dos_info(task_directory: str, plot_dos: bool = True, smooth: int = 0) -> dict:
    """提取态密度计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_dos_info(task_directory)
            if plot_dos and result and isinstance(result, dict):
                local_files = result.get('local_files', {}) or {}
                vasprun_path = local_files.get('vasprun.xml') or local_files.get('vasprun')

                if vasprun_path and os.path.exists(vasprun_path):
                    res = plot_vasp_dos_analysis(vasprun_path, smooth=smooth)

                    image = res.get('Image') if isinstance(res, dict) else None
                    payload = {k: v for k, v in res.items() if k != 'Image'} if isinstance(res, dict) else {}
                    if not res.get('error'):
                        result.update({
                            'image_url': image,
                            'plot_info': payload,
                            'message': '绘图成功',
                        })
                    else:
                        result.update({
                            'image_url': image,
                            'plot_info': payload,
                            'message': '绘图失败',
                        })
                else:
                    result.setdefault('warnings', []).append('vasprun.xml文件缺失，无法绘图。')
            return result
    except Exception as e:
        return {'error': str(e), 'message': '提取任务结果失败'}


# ============ MCP 工具 ============



"""
!!!
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
!!!
"""

# ----- 基础工具 -----
# @mcp.tool()
# async def get_time() -> dict:
#     """获取当前时间，返回格式：YYYY-MM-DD HH:MM:SS"""
#     args = {}
#     result = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
#     return {"args": args, "returns": result}


# @mcp.tool()
# async def get_material_project_page(material_id: str) -> dict:
#     """
#     获取指定材料的Material Project页面链接
    
#     Args:
#         material_id: 材料ID，如 "mp-1234"
    
#     Returns:
#         dict: 包含 material_id, url, message, error(可选)
#     """
#     args = {"material_id": material_id}
#     if not material_id:
#         return {"args": args, "returns": {"error": "材料ID不能为空", "message": "请提供有效的材料ID"}}
    
#     url = f"https://next-gen.materialsproject.org/materials/{material_id}/"
#     return {"args": args, "returns": {"url": url}}


@mcp.tool()
async def read_file(file_path: str) -> dict:
    """
    读取mcp服务器的文件
    
    Args:
        file_path: 文件的绝对路径或相对路径
    
    Returns:
        dict: 包含 success, content, file_path, error(可选)
    """
    args = {"file_path": file_path}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {"args": args, "returns": {
            "success": True,
            "content": content,
        }}
    except FileNotFoundError:
        return {"args": args, "returns": {"success": False, "error": f"文件未找到: {file_path}"}}
    except PermissionError:
        return {"args": args, "returns": {"success": False, "error": f"权限不足: {file_path}"}}
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": f"读取文件时出错: {str(e)}"}}


# ----- OQMD 数据库工具 -----
@mcp.tool()
async def search_materials_from_oqmd(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_max: float = 0.1,
    limit: int = 20,
    num_elements_min: Optional[int] = None,
    num_elements_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    在OQMD数据库搜索材料

    Args:
        elements: 元素列表，如 ["Li", "Fe", "O"] 或 ["Si"]
        band_gap_min: 最小带隙值（eV），如 0.0
        band_gap_max: 最大带隙值（eV），如 3.0
        stability_max: 最大稳定性值（eV/atom），默认 0.1
        limit: 返回结果数量限制，默认 20
        num_elements_min: 最小元素种类数
        num_elements_max: 最大元素种类数

    Returns:
        Dict: OQMD搜索结果
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if band_gap_min is not None:
        args["band_gap_min"] = band_gap_min
    if band_gap_max is not None:
        args["band_gap_max"] = band_gap_max
    if stability_max != 0.1:
        args["stability_max"] = stability_max
    if limit != 20:
        args["limit"] = limit
    if num_elements_min is not None:
        args["num_elements_min"] = num_elements_min
    if num_elements_max is not None:
        args["num_elements_max"] = num_elements_max

    filter_parts = []

    if elements:
        element_set = ",".join(elements)
        filter_parts.append(f"element_set={element_set}")

    if band_gap_min is not None:
        filter_parts.append(f"band_gap>={band_gap_min}")

    if band_gap_max is not None:
        filter_parts.append(f"band_gap<={band_gap_max}")

    if stability_max is not None:
        filter_parts.append(f"stability<={stability_max}")

    if num_elements_min is not None:
        filter_parts.append(f"ntypes>={num_elements_min}")
    if num_elements_max is not None:
        filter_parts.append(f"ntypes<={num_elements_max}")
    
    filter_expr = " AND ".join(filter_parts) if filter_parts else None
    
    fields = ["name", "entry_id", "band_gap", "delta_e", "stability", "spacegroup", "ntypes"]
    
    result = oqmd.search_oqmd(
        fields=fields,
        filter_expr=filter_expr,
        limit=limit,
        offset=0,
        sort_by="stability",
        desc=False
    )
    return {"args": args, "returns": result}


@mcp.tool()
async def get_material_structure_from_oqmd(
    entry_id: int,
    mode: str = "conventional",
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False,
    analyze: str = "off"
) -> dict | list:
    """
    在OQMD数据库获取指定材料的结构

    Args:
        entry_id: OQMD材料条目ID（整数）
        mode: 晶胞类型，"conventional"（常规晶胞，默认）或 "primitive"（原胞）
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        download: 是否下载CIF文件，默认 False
        analyze: 结构分析详细程度 — "off" (不分析), "brief" (仅统计), "normal" (统计+精简列表), "full" (完整列表)

    Returns:
        dict | list: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {"entry_id": entry_id, "mode": mode, "get_sites": get_sites, "get_plot": get_plot, "download": download, "analyze": analyze}
    res = oqmd.parse_poscar_with_pymatgen(entry_id, mode)
    message = []
    if res["success"]:
        structure = res["structure"]
        lattice = structure.lattice
        space_group_info = structure.get_space_group_info()
        formula = structure.formula
        reduced_formula = structure.composition.reduced_formula
        structure_info = {
            'formula': formula,
            'reduced_formula': reduced_formula,
            'space_group_symbol': space_group_info[0] if space_group_info else "未知",
            'space_group_number': space_group_info[1] if space_group_info else "未知",
            'lattice_parameters': {
                'a': round(lattice.a, 4),
                'b': round(lattice.b, 4),
                'c': round(lattice.c, 4),
                'alpha': round(lattice.alpha, 2),
                'beta': round(lattice.beta, 2),
                'gamma': round(lattice.gamma, 2),
                'volume': round(lattice.volume, 4)
            },
            'number_of_sites': len(structure),
            'density': round(structure.density, 4),
            'is_ordered': structure.is_ordered,
        }
        structure_info = _clean_numpy(structure_info)
        if analyze != "off":
            structure_info['analysis'] = analyze_structure(structure, detail=analyze)
            message.append("结构分析（Wyckoff位置/键长/键角）已包含在返回结果中")
        if get_sites:
            structure_info['sites'] = [{
                'element': site.species_string,
                'fractional_coordinates': [round(coord, 4) for coord in site.frac_coords],
            } for site in structure.sites]
            message.append(f"材料 {entry_id} 的原子位点信息已包含在返回结果中")
        if download:
            CifWriter(structure).write_file(f"cifs/{reduced_formula}-oqmd-{entry_id}.cif")
            print(f"获取材料 {entry_id} 的晶体结构成功，已保存为cif文件")
            message.append(f"材料 {entry_id} 的晶体结构已保存为cif文件，路径为'cifs/{reduced_formula}-oqmd-{entry_id}.cif'")
        if get_plot:
            structure_url = visualize_structure(structure)
            message.append("生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图")
            message.append(f"3d_image_url: {structure_url}")
            res = get_structure_plot(structure)
            if not res["error"]:
                image = res["Image"]
                return {"args": args, "returns": {"image_url": image, "structure_dict": structure_info, "message": message}}
            else:
                message.append(res["error"])
                return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}

        return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}
    else:
        return {"args": args, "returns": {"error": res["error"], "message": "构建晶体结构失败"}}


# ----- AFLOW 数据库工具 -----
@mcp.tool()
async def search_materials_from_aflow(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_max: Optional[float] = None,
    limit: int = 20,
    num_elements_min: Optional[int] = None,
    num_elements_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    在AFLOW数据库搜索材料

    Args:
        elements: 元素列表，如 ["Li", "Fe", "O"] 或 ["Si"]
        band_gap_min: 最小带隙值（eV）
        band_gap_max: 最大带隙值（eV）
        stability_max: 最大生成焓（eV/atom），越负越稳定，默认不限制
        limit: 返回结果数量限制，默认 20
        num_elements_min: 最小元素种类数
        num_elements_max: 最大元素种类数

    Returns:
        Dict: AFLOW搜索结果
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if band_gap_min is not None:
        args["band_gap_min"] = band_gap_min
    if band_gap_max is not None:
        args["band_gap_max"] = band_gap_max
    if stability_max is not None:
        args["stability_max"] = stability_max
    if limit != 20:
        args["limit"] = limit
    if num_elements_min is not None:
        args["num_elements_min"] = num_elements_min
    if num_elements_max is not None:
        args["num_elements_max"] = num_elements_max

    result = aflow_search.search_aflow(
        elements=elements,
        band_gap_min=band_gap_min,
        band_gap_max=band_gap_max,
        stability_max=stability_max,
        num_elements_min=num_elements_min,
        num_elements_max=num_elements_max,
        limit=limit,
    )
    return {"args": args, "returns": result}


@mcp.tool()
async def get_material_structure_from_aflow(
    auid: str,
    aurl: str,
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False,
    analyze: str = "off"
) -> dict | list:
    """
    在AFLOW数据库获取指定材料的结构

    Args:
        auid: AFLOW唯一标识符，如 "aflow:0fd4cd5b650d72e0"
        aurl: AFLOW数据路径，如 "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/ORC/Li1O2_ICSD_180561"
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        download: 是否下载CIF文件，默认 False
        analyze: 结构分析详细程度 — "off" (不分析), "brief" (仅统计), "normal" (统计+精简列表), "full" (完整列表)

    Returns:
        dict | list: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {"auid": auid, "aurl": aurl, "get_sites": get_sites, "get_plot": get_plot, "download": download, "analyze": analyze}

    try:
        path = aurl.split(":", 1)[1] if ":" in aurl else aurl
        url = f"https://aflowlib.duke.edu/{path}/CONTCAR.relax"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return {"args": args, "returns": {"error": f"Failed to fetch structure: HTTP {resp.status_code}"}}

        structure = Structure.from_str(resp.text, fmt="poscar")
    except Exception as e:
        return {"args": args, "returns": {"error": f"结构解析失败: {str(e)}", "message": "构建晶体结构失败"}}

    lattice = structure.lattice
    space_group_info = structure.get_space_group_info()
    formula = structure.formula
    reduced_formula = structure.composition.reduced_formula
    structure_info = {
        'formula': formula,
        'reduced_formula': reduced_formula,
        'space_group_symbol': space_group_info[0] if space_group_info else "未知",
        'space_group_number': space_group_info[1] if space_group_info else "未知",
        'lattice_parameters': {
            'a': round(lattice.a, 4),
            'b': round(lattice.b, 4),
            'c': round(lattice.c, 4),
            'alpha': round(lattice.alpha, 2),
            'beta': round(lattice.beta, 2),
            'gamma': round(lattice.gamma, 2),
            'volume': round(lattice.volume, 4)
        },
        'number_of_sites': len(structure),
        'density': round(structure.density, 4),
        'is_ordered': structure.is_ordered,
    }
    structure_info = _clean_numpy(structure_info)

    message = []
    if analyze != "off":
        structure_info['analysis'] = analyze_structure(structure, detail=analyze)
        message.append("结构分析（Wyckoff位置/键长/键角）已包含在返回结果中")
    if get_sites:
        structure_info['sites'] = [{
            'element': site.species_string,
            'fractional_coordinates': [round(coord, 4) for coord in site.frac_coords],
        } for site in structure.sites]
        message.append(f"材料 {auid} 的原子位点信息已包含在返回结果中")

    if download:
        CifWriter(structure).write_file(f"cifs/{reduced_formula}-aflow-{auid.replace(':', '_')}.cif")
        message.append(f"材料 {auid} 的晶体结构已保存为cif文件")

    if get_plot:
        structure_url = visualize_structure(structure)
        message.append("生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图")
        message.append(f"3d_image_url: {structure_url}")
        res = get_structure_plot(structure)
        if not res["error"]:
            image = res["Image"]
            return {"args": args, "returns": {"image_url": image, "structure_dict": structure_info, "message": message}}
        else:
            message.append(res["error"])
            return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}

    return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}


# ----- Alexandria 数据库工具 -----
@mcp.tool()
async def search_materials_from_alexandria(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    hull_distance_max: Optional[float] = None,
    limit: int = 20,
    num_elements_min: Optional[int] = None,
    num_elements_max: Optional[int] = None,
    functional: str = "pbe"
) -> Dict[str, Any]:
    """
    在 Alexandria 材料数据库搜索材料 (OPTIMADE v1.1 接口)

    Args:
        elements: 元素列表，如 ["Li", "Fe", "O"] 或 ["Si"]
        band_gap_min: 最小带隙值（eV）
        band_gap_max: 最大带隙值（eV）
        hull_distance_max: 最大 hull distance（eV/atom），0 = 在凸包上
        limit: 返回结果数量限制，默认 20
        num_elements_min: 最小元素种类数
        num_elements_max: 最大元素种类数
        functional: 交换关联泛函 — "pbe" (默认), "pbesol", "scan"

    Returns:
        Dict: Alexandria 搜索结果，包含 data 和 meta
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if band_gap_min is not None:
        args["band_gap_min"] = band_gap_min
    if band_gap_max is not None:
        args["band_gap_max"] = band_gap_max
    if hull_distance_max is not None:
        args["hull_distance_max"] = hull_distance_max
    if limit != 20:
        args["limit"] = limit
    if num_elements_min is not None:
        args["num_elements_min"] = num_elements_min
    if num_elements_max is not None:
        args["num_elements_max"] = num_elements_max
    if functional != "pbe":
        args["functional"] = functional

    filter_parts = []
    if elements:
        elem_str = ", ".join(f'"{e}"' for e in elements)
        filter_parts.append(f"elements HAS ALL {elem_str}")
    if band_gap_min is not None:
        filter_parts.append(f"_alexandria_band_gap >= {band_gap_min}")
    if band_gap_max is not None:
        filter_parts.append(f"_alexandria_band_gap <= {band_gap_max}")
    if hull_distance_max is not None:
        filter_parts.append(f"_alexandria_hull_distance <= {hull_distance_max}")
    if num_elements_min is not None:
        filter_parts.append(f"nelements >= {num_elements_min}")
    if num_elements_max is not None:
        filter_parts.append(f"nelements <= {num_elements_max}")

    filter_str = " AND ".join(filter_parts) if filter_parts else None

    base_url = f"https://alexandria.icams.rub.de/{functional}/v1/structures"
    params = {"page_limit": limit}
    if filter_str:
        params["filter"] = filter_str

    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"args": args, "returns": {"error": f"Alexandria 查询失败: {str(e)}"}}

    materials = []
    for struct in data.get("data", []):
        attrs = struct.get("attributes", {})
        mat = {
            "id": struct.get("id"),
            "formula": attrs.get("chemical_formula_reduced", "Unknown"),
            "formula_descriptive": attrs.get("chemical_formula_descriptive", ""),
            "elements": attrs.get("elements", []),
            "nelements": attrs.get("nelements", 0),
            "nsites": attrs.get("nsites", 0),
            "band_gap": attrs.get("_alexandria_band_gap"),
            "band_gap_direct": attrs.get("_alexandria_band_gap_direct"),
            "formation_energy_per_atom": attrs.get("_alexandria_formation_energy_per_atom"),
            "energy": attrs.get("_alexandria_energy"),
            "energy_corrected": attrs.get("_alexandria_energy_corrected"),
            "hull_distance": attrs.get("_alexandria_hull_distance"),
            "phase_separation_energy": attrs.get("_alexandria_phase_separation_energy"),
            "magnetization": attrs.get("_alexandria_magnetization"),
            "dos_ef": attrs.get("_alexandria_dos_ef"),
            "space_group": attrs.get("_alexandria_space_group"),
            "xc_functional": attrs.get("_alexandria_xc_functional"),
            "decomposition": attrs.get("_alexandria_decomposition"),
        }
        mat = _clean_numpy(mat)
        materials.append(mat)

    return {
        "args": args,
        "returns": {
            "data": materials,
            "meta": {"data_available": data.get("meta", {}).get("data_available", len(materials))}
        }
    }


@mcp.tool()
async def get_material_structure_from_alexandria(
    material_id: str,
    functional: str = "pbe",
    get_sites: bool = False,
    get_plot: bool = False,
    analyze: str = "off"
) -> dict:
    """
    在 Alexandria 数据库获取指定材料的结构

    Args:
        material_id: Alexandria 材料ID，如 "agm010193661"
        functional: 交换关联泛函 — "pbe" (默认), "pbesol", "scan"
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        analyze: 结构分析详细程度 — "off" (不分析), "brief" (仅统计), "normal" (统计+精简列表), "full" (完整列表)

    Returns:
        dict: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {"material_id": material_id, "functional": functional, "get_sites": get_sites, "get_plot": get_plot, "analyze": analyze}

    base_url = f"https://alexandria.icams.rub.de/{functional}/v1/structures"
    params = {"filter": f'id="{material_id}"'}

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"args": args, "returns": {"error": f"Alexandria 结构获取失败: {str(e)}"}}

    structures = data.get("data", [])
    if not structures:
        return {"args": args, "returns": {"error": f"未找到材料: {material_id}"}}

    attrs = structures[0].get("attributes", {})
    lattice_vectors = attrs.get("lattice_vectors")
    cartesian_sites = attrs.get("cartesian_site_positions")
    species_at_sites = attrs.get("species_at_sites")

    if not lattice_vectors or not cartesian_sites or not species_at_sites:
        return {"args": args, "returns": {"error": "结构数据不完整"}}

    try:
        structure = Structure(
            lattice=lattice_vectors,
            species=species_at_sites,
            coords=cartesian_sites,
            coords_are_cartesian=True
        )
    except Exception as e:
        return {"args": args, "returns": {"error": f"结构解析失败: {str(e)}"}}

    formula = structure.formula
    reduced_formula = structure.composition.reduced_formula
    space_group_info = structure.get_space_group_info()
    structure_info = {
        'formula': formula,
        'reduced_formula': reduced_formula,
        'space_group_symbol': space_group_info[0] if space_group_info else "未知",
        'space_group_number': space_group_info[1] if space_group_info else "未知",
        'lattice_parameters': {
            'a': round(structure.lattice.a, 4),
            'b': round(structure.lattice.b, 4),
            'c': round(structure.lattice.c, 4),
            'alpha': round(structure.lattice.alpha, 2),
            'beta': round(structure.lattice.beta, 2),
            'gamma': round(structure.lattice.gamma, 2),
            'volume': round(structure.lattice.volume, 4)
        },
        'number_of_sites': len(structure),
        'density': round(structure.density, 4),
        'is_ordered': structure.is_ordered,
    }
    structure_info = _clean_numpy(structure_info)

    message = []
    if analyze != "off":
        structure_info['analysis'] = analyze_structure(structure, detail=analyze)
        message.append("结构分析（Wyckoff位置/键长/键角）已包含在返回结果中")
    if get_sites:
        structure_info['sites'] = [{
            'element': site.species_string,
            'fractional_coordinates': [round(coord, 4) for coord in site.frac_coords],
        } for site in structure.sites]
        message.append(f"材料 {material_id} 的原子位点信息已包含在返回结果中")

    if get_plot:
        structure_url = visualize_structure(structure)
        message.append("生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图")
        message.append(f"3d_image_url: {structure_url}")
        res = get_structure_plot(structure)
        if not res["error"]:
            image = res["Image"]
            return {"args": args, "returns": {"image_url": image, "structure_dict": structure_info, "message": message}}
        else:
            message.append(res["error"])
            return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}

    return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}


# ----- Material Project 工具 -----
@mcp.tool()
async def search_materials_from_mp(
    elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    chemsys: str | list[str] | None = None,
    band_gap: tuple[float, float] | None = None,
    num_elements: tuple[int, int] | None = None,
    formula: str | list[str] | None = None,
    chunk_size: int | None = 25,
    energy_above_hull_min: float | None = None,
    energy_above_hull_max: float | None = None
) -> dict:
    """
    Material Project数据查询工具

    Args:
        elements: 包含的元素列表，如 ["Li", "Fe", "O"]
        exclude_elements: 排除的元素列表，如 ["C", "N"]
        chemsys: 化学系统，如 "Li-Fe-O" 或 ["Li-Fe-O", "Na-Cl"]
        band_gap: 带隙范围元组，如 (0.0, 3.0)
        num_elements: 元素数量范围，如 (2, 4)
        formula: 化学式，如 "LiFeO2" 或 ["LiFeO2", "NaCl"]
        chunk_size: 每块返回数量，默认25
        energy_above_hull_min: 最小形成能 (eV/atom)，默认不限制
        energy_above_hull_max: 最大形成能 (eV/atom)，默认不限制

    Returns:
        list[dict]: 材料列表（按 energy_above_hull 升序，最稳定的排在前面）
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if exclude_elements is not None:
        args["exclude_elements"] = exclude_elements
    if chemsys is not None:
        args["chemsys"] = chemsys
    if band_gap is not None:
        args["band_gap"] = band_gap
    if num_elements is not None:
        args["num_elements"] = num_elements
    if formula is not None:
        args["formula"] = formula
    if chunk_size is not None and chunk_size != 25:
        args["chunk_size"] = chunk_size
    if energy_above_hull_min is not None:
        args["energy_above_hull_min"] = energy_above_hull_min
    if energy_above_hull_max is not None:
        args["energy_above_hull_max"] = energy_above_hull_max
    
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("API密钥未设置")

    # 检查是否提供了任何筛选条件
    has_filter = any([
        elements, exclude_elements, chemsys, band_gap, num_elements,
        formula, energy_above_hull_min is not None, energy_above_hull_max is not None,
    ])
    if not has_filter:
        return {"args": args, "returns": {
            "error": "未提供任何筛选条件，搜索范围过大。请提供至少一个筛选条件：elements（元素列表，如 ['Li', 'Fe', 'O']）、chemsys（化学系统，如 'Li-Fe-O'）、formula（化学式，如 'LiFeO2'）、band_gap（带隙范围，如 (0, 3)）或 num_elements（元素数量范围，如 (2, 4)）"
        }}

    try:
        with MPRester(API_KEY) as mpr:
            search_kwargs = {}
            
            if elements:
                search_kwargs["elements"] = elements
            if exclude_elements:
                search_kwargs["exclude_elements"] = exclude_elements
            if chemsys:
                search_kwargs["chemsys"] = chemsys
            if band_gap:
                search_kwargs["band_gap"] = band_gap
            if num_elements:
                search_kwargs["num_elements"] = num_elements
            if formula:
                search_kwargs["formula"] = formula

            search_kwargs["fields"] = ["material_id", "formula_pretty", "band_gap", "symmetry", "energy_above_hull", "is_stable"]
            chunk_sz = chunk_size if chunk_size else 25

            results = mpr.materials.summary.search(**search_kwargs, chunk_size=chunk_sz, num_chunks=1)
            # 在 Python 端按 energy_above_hull 过滤和排序
            if energy_above_hull_min is not None and energy_above_hull_max is not None:
                results = [r for r in results
                           if r.energy_above_hull is not None
                           and energy_above_hull_min <= r.energy_above_hull <= energy_above_hull_max]
            # 按稳定性排序: energy_above_hull 越小越稳定
            results = sorted(results, key=lambda r: r.energy_above_hull or 999)
            print(f"查询到 {len(results)} 个材料")
            data = {
                "data": [
                    {
                        "formula": r.formula_pretty,
                        "material_id": r.material_id,
                        "symmetry": r.symmetry,
                        "band_gap": r.band_gap,
                        "energy_above_hull": r.energy_above_hull,
                        "is_stable": r.is_stable
                    } for r in results
                ]
            }
        return {"args": args, "returns": data}
    except Exception as e:
        import traceback
        print(f"[ERROR] search_materials_from_mp: {e}")
        print(traceback.format_exc())
        return {"args": args, "returns": {"error": str(e), "message": "查询材料数据失败"}}


@mcp.tool()
async def get_band_gap(material_id: str) -> dict:
    """
    获取指定材料的带隙值(Material Project)
    
    Args:
        material_id: 材料ID，如 "mp-1234"
    
    Returns:
        dict: 包含 material_id, band_gap, formula, error(可选)
    """
    args = {"material_id": material_id}
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    try:
        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(
                material_ids=material_id,
                fields=["band_gap", "formula_pretty"]
            )
            if not results:
                raise ValueError(f"未找到材料ID为 {material_id} 的材料")
            else:
                print(f"获取材料 {material_id} 的带隙值成功")
            band_gap = results[0].band_gap
            formula = results[0].formula_pretty
        return {"args": args, "returns": {"band_gap": band_gap}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": f"获取材料 {material_id} 的带隙值失败"}}


@mcp.tool()
async def get_material_structure_from_mp(
    material_id: str,
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False,
    analyze: str = "off"
) -> dict | list:
    """
    在Material Project上获取指定材料的晶体结构数据

    Args:
        material_id: 材料ID，如 "mp-1234"
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        download: 是否下载CIF文件，默认 False
        analyze: 结构分析详细程度 — "off" (不分析), "brief" (仅统计), "normal" (统计+精简列表), "full" (完整列表)
    
    Returns:
        dict | list: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {"material_id": material_id, "get_sites": get_sites, "get_plot": get_plot, "download": download, "analyze": analyze}
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    os.makedirs("cifs", exist_ok=True)
    os.makedirs("cifs/images", exist_ok=True)
    message = []
    try:
        with MPRester(API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=True)
            lattice = structure.lattice
            space_group_info = structure.get_space_group_info()
            formula = structure.formula
            reduced_formula = structure.composition.reduced_formula
            structure_info = {
                'formula': formula,
                'reduced_formula': reduced_formula,
                'space_group_symbol': space_group_info[0] if space_group_info else "未知",
                'space_group_number': space_group_info[1] if space_group_info else "未知",
                'lattice_parameters': {
                    'a': round(lattice.a, 4),
                    'b': round(lattice.b, 4),
                    'c': round(lattice.c, 4),
                    'alpha': round(lattice.alpha, 2),
                    'beta': round(lattice.beta, 2),
                    'gamma': round(lattice.gamma, 2),
                    'volume': round(lattice.volume, 4)
                },
                'number_of_sites': len(structure),
                'density': round(structure.density, 4),
                'is_ordered': structure.is_ordered,
            }
            structure_info = _clean_numpy(structure_info)
            if analyze != "off":
                structure_info['analysis'] = analyze_structure(structure, detail=analyze)
                message.append("结构分析（Wyckoff位置/键长/键角）已包含在返回结果中")
            message.append(f"材料 {material_id} 的晶体结构信息: formula={formula}, space_group={space_group_info[0] if space_group_info else '未知'}")
            if get_sites:
                sites_data = [{
                    'element': site.species_string,
                    'fractional_coordinates': [round(coord, 4) for coord in site.frac_coords],
                } for site in structure.sites]
                structure_info['sites'] = sites_data
                structure_info['sites_count'] = len(structure.sites)
                message.append(f"材料 {material_id} 的原子位点信息已包含在返回结果中，共{len(structure.sites)}个)")
            if download:
                CifWriter(structure).write_file(f"cifs/{reduced_formula}-{material_id}.cif")
                print(f"获取材料 {material_id} 的晶体结构成功，已保存为cif文件")
                message.append(f"材料 {material_id} 的晶体结构已保存为cif文件，路径为'cifs/{reduced_formula}-{material_id}.cif'")   
            if get_plot:
                structure_url = visualize_structure(structure)
                message.append("生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图")
                message.append(f"3d_image_url: {structure_url}")
                res = get_structure_plot(structure)
                if not res["error"]:
                    image = res["Image"]
                    return {
                        "args": args,
                        "returns": {"image_url": image, "3d_image_url": structure_url, "structure_dict": structure_info, "message": message}
                    }
                else:
                    message.append(res["error"])
                    return {
                        "args": args,
                        "returns": {"3d_image_url": structure_url, "structure_dict": structure_info, "message": message}
                    }

        return {"args": args, "returns": {"structure_dict": structure_info, "message": message}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": f"获取材料 {material_id} 的晶体结构失败"}}


@mcp.tool()
async def get_material_all_infomation_by_id(material_id: str) -> dict:
    """获取Material Project指定材料的所有信息"""
    args = {"material_id": material_id}
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")

    try:
        with MPRester(API_KEY) as mpr:
            with mpr.materials as materials:
                material = materials.search(material_ids=material_id)
                if not material:
                    raise ValueError(f"未找到材料ID为 {material_id} 的材料")
                else:
                    print(f"获取材料 {material_id} 的所有信息成功")
            material_dict = material[0]
        return {"args": args, "returns": material_dict}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": f"获取材料 {material_id} 的所有信息失败"}}


# ----- 结构建模工具 -----
@mcp.tool()
async def build_structure(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    elements: list[str],
    frac_coord: list[list[float]],
    scaling_matrix: int | list = 1,
    save_to_cif: bool = False,
    add_to_database: str = None,
    analyze: str = "off",
) -> dict | list:
    """
    构建晶体结构并保存为CIF文件，生成晶体结构图

    Args:
        a: 晶格参数a（Å）
        b: 晶格参数b（Å）
        c: 晶格参数c（Å）
        alpha: 晶格角alpha（度）
        beta: 晶格角beta（度）
        gamma: 晶格角gamma（度）
        elements: 元素符号列表，如 ["Na", "Cl"]
        frac_coord: 分数坐标列表，如 [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        scaling_matrix: 超胞，默认整数（int）：表示在 a, b, c 三个方向进行相同的扩胞。例如 scaling_matrix=2表示构建 2×2×2 的超胞。
                    列表（list）：长度为 3 的列表，分别表示 a, b, c 方向的扩胞倍数。例如 scaling_matrix=[2, 1, 1]表示构建 2×1×1 的超胞。
        save_to_cif: 是否保存为CIF文件，默认False
        add_to_database: 数据库文件名，如添加则保存到该数据库
        analyze: 结构分析详细程度 — "off" (不分析), "brief" (仅统计), "normal" (统计+精简列表), "full" (完整列表)

    Returns:
        dict | list: 包含 image, 3d_image_url, message, analysis(可选), error(可选)
    """
    args = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma, "elements": elements, "frac_coord": frac_coord, "scaling_matrix": scaling_matrix, "save_to_cif": save_to_cif, "analyze": analyze}
    if add_to_database is not None:
        args["add_to_database"] = add_to_database
    
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        structure = Structure(lattice, elements, frac_coord)
        structure = structure.make_supercell(scaling_matrix=scaling_matrix)
        formula = structure.composition.reduced_formula
        os.makedirs("custom_structures", exist_ok=True)
        os.makedirs("custom_structures/images", exist_ok=True)
        message = []
        current_date = datetime.now().strftime("%Y%m%d%H%M")
        if save_to_cif:
            CifWriter(structure).write_file(f"custom_structures/{formula}_custom_{current_date}.cif")
            message.append(f"自定义晶体结构已保存为 ./custom_structures/{formula}_custom_{current_date}.cif")
        structure_url = visualize_structure(structure)
        message.append("3d晶体结构可视化交互式网页，请点击查看晶体结构图")

        analysis_data = None
        if analyze != "off":
            analysis_data = analyze_structure(structure, detail=analyze)
            message.append("结构分析（Wyckoff位置/键长/键角）已包含在返回结果中")

        if add_to_database:
            db = databasemanage.DatabaseManager(add_to_database)
            db.add_material(formula=formula, structure=structure, band_gap=None, material_id=None)
            db.close()
            message.append(f"自定义晶体结构已添加到数据库 {add_to_database}")

        res = get_structure_plot(structure)
        if not res["error"]:
            image = res["Image"]
            returns = {"image": image, "3d_image_url": structure_url, "message": message}
            if analysis_data:
                returns["analysis"] = analysis_data
            return {"args": args, "returns": returns}
        else:
            message.append(res["error"])
            returns = {"3d_image_url": structure_url, "message": message}
            if analysis_data:
                returns["analysis"] = analysis_data
            return {"args": args, "returns": returns}

    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "构建晶体结构失败"}}


# ----- VASP 任务管理工具 -----
@mcp.tool()
async def create_task(formula: str, cif_path: str) -> dict:
    """
    在远程服务器上创建任务文件夹并上传CIF文件
    
    Args:
        formula: 材料化学式，如 "LiFeO2"
        cif_path: 本地CIF文件路径
    
    Returns:
        dict: 包含 message, task_directory, error(可选)
    """
    args = {"formula": formula, "cif_path": cif_path}
    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.create_task(formula, cif_path, base_dir)
                if result:
                    break
            if result:
                return {"args": args, "returns": {"message": f"任务目录已创建并上传CIF文件", "task_directory": result}}
            else:
                return {"args": args, "returns": {"error": "任务创建失败", "message": "请再试一次"}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "任务创建失败"}}


@mcp.tool()
async def list_task_directories() -> dict:
    """
    列出远程服务器上的所有任务目录
    """
    args = {}
    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.get_task_directories(base_dir)
                if result:
                    break
            if result:
                return {"args": args, "returns": {"task_directories": result}}
            else:
                return {"args": args, "returns": {"error": "获取任务目录失败", "message": "请检查服务器连接是否正常"}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "获取任务目录失败"}}


@mcp.tool()
async def check_squeue() -> dict:
    """
    检查远程服务器上的任务队列
    """
    args = {}
    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.check_squeue()
                if result:
                    break
            if result:
                return {"args": args, "returns": {"squeue": result}}
            else:
                return {"args": args, "returns": {"error": "检查任务队列失败", "message": "请检查服务器连接是否正常"}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "检查任务队列失败"}}


@mcp.tool()
async def execute_command(command: str) -> dict:
    """
    在计算服务器上执行linux命令
    
    Args:
        command: 要执行的Linux命令，如 "ls -la"
    
    Returns:
        dict: 命令执行结果或错误信息
    """
    args = {"command": command}
    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}
    try:
        with connection as vasp_task:
            result = vasp_task.execute_command(command)
            return {"args": args, "returns": result}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "命令提交或执行失败"}}


@mcp.tool()
async def extract_file(file_path: str) -> dict:
    """
    从计算服务器上提取一个文件，并提供下载的URL
    
    Args:
        file_path: 远程服务器上的文件路径
    
    Returns:
        dict: 包含以下字段:
            - local_file: 本地保存的文件路径
            - download_url: 文件下载URL
            - error: 错误信息（如果有）
            - message: 操作结果消息
    """
    args = {"file_path": file_path}
    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}
    try:
        with connection as vasp_task:
            result = vasp_task.extract_file(file_path=file_path)
            download_url = matfileserver.add_image_file(result["local_file"])
            result["download_url"] = download_url
            return {"args": args, "returns": result}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "提取文件失败"}}


@mcp.tool()
async def create_mission(task_directory: str, mission: str) -> dict:
    """
    创建计算任务的输入文件（POSCAR、INCAR、POTCAR、KPOINTS），但不提交计算
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）
    
    Returns:
        dict: 包含 success, mission, task_directory, raw_result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission}
    mission = mission.lower().strip()
    method_map = {
        "relax": "create_relax_mission",
        "scf": "create_scf_mission",
        "band": "create_band_mission",
        "dos": "create_dos_mission"
    }

    if mission not in method_map:
        return {"args": args, "returns": {"success": False, "error": f"未知的计算类型: {mission}，可选: {list(method_map.keys())}"}}

    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}

    try:
        with connection as vasp_task:
            method_name = method_map[mission]
            method = getattr(vasp_task, method_name)
            result = method(task_directory)
            
            success = result.get("status") == "ok" or "error" not in result
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "raw_result": result
            }
            
            if not success:
                response["error"] = result.get("error") or result.get("message") or "创建任务失败"
            
            return {"args": args, "returns": response}
            
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
async def submit_mission(task_directory: str, mission: str) -> dict:
    """
    提交已准备好的计算任务
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）
    
    Returns:
        dict: 包含 success, mission, task_directory, job_id(可选), message, raw_result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission}
    mission = mission.lower().strip()
    method_map = {
        "relax": "submit_relax_calculation",
        "scf": "submit_scf_calculation",
        "band": "submit_band_calculation",
        "dos": "submit_dos_calculation"
    }

    if mission not in method_map:
        return {"args": args, "returns": {"success": False, "error": f"未知的计算类型: {mission}，可选: {list(method_map.keys())}"}}

    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}

    try:
        with connection as vasp_task:
            method_name = method_map[mission]
            method = getattr(vasp_task, method_name)
            result = method(task_directory)
            
            success = result.get("status") == "ok" or "error" not in result
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "job_id": result.get("job_id"),
                "raw_result": result
            }
            
            if success:
                response["message"] = f"{mission}计算任务提交成功"
                if result.get("job_id"):
                    response["message"] += f"，作业ID: {result['job_id']}"
                    response["message"] += f"使用工具 extract_result {task_directory} {mission} 来提取计算结果"
            else:
                response["error"] = result.get("error") or result.get("message") or "提交任务失败"
            
            return {"args": args, "returns": response}
            
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
async def modify_incar(task_directory: str, mission: str, read: bool, write: str = None) -> dict:
    """
    读写修改计算任务的INCAR文件
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"、"scf"、"band"、"dos"
        read: 是否为读取模式，True为读取，False为写入
        write: 要写入的INCAR参数（JSON格式字符串），仅在read=False时使用，如: '{"ENCUT": 520}'
    
    Returns:
        dict: 包含 success, mission, task_directory, read_mode, incar_params(可选), updated_params(可选), message, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission, "read": read}
    if write is not None:
        args["write"] = write
    
    mission = mission.lower().strip()
    
    new_params = None
    if not read and write:
        try:
            new_params = json.loads(write)
            if not isinstance(new_params, dict):
                return {"args": args, "returns": {"success": False, "error": "write参数必须是JSON对象（字典）"}}
        except Exception as e:
            return {"args": args, "returns": {"success": False, "error": f"解析write参数失败: {str(e)}"}}

    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}

    try:
        with connection as vasp_task:
            result = vasp_task.modify_incar_file(
                task_directory=task_directory,
                mission=mission,
                read_mode=read,
                new_params=new_params
            )
            
            success = result.get("status") == "ok"
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "read_mode": read,
                "raw_result": result
            }
            
            if success:
                if read:
                    response["incar_params"] = result.get("incar_params", {})
                    response["message"] = f"成功读取{mission}任务的INCAR参数"
                else:
                    response["message"] = result.get("message", "INCAR文件更新成功")
                    response["updated_params"] = result.get("updated_params", [])
            else:
                response["error"] = result.get("error") or result.get("message") or "操作失败"
            
            return {"args": args, "returns": response}
            
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
def extract_result(task_directory: str, mission: str, plot: bool = True, smooth: int = 0) -> dict:
    """
    提取计算任务的结果

    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）
        plot: 是否生成图表，默认True
        smooth: DOS 曲线 Savitzky-Golay 平滑窗口大小，0 则不平滑 (仅对 mission="dos" 有效)

    Returns:
        dict: 包含 success, mission, task_directory, result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission, "plot": plot, "smooth": smooth}
    mission = mission.lower().strip()
    method_map = {
        "relax": lambda: extract_relax_info(task_directory, get_plot=plot, visualize=plot),
        "scf": lambda: extract_scf_info(task_directory),
        "band": lambda: extract_band_info(task_directory, plot_band=plot),
        "dos": lambda: extract_dos_info(task_directory, plot_dos=plot, smooth=smooth),
    }

    if mission not in method_map:
        return {"args": args, "returns": {"success": False, "error": f"未知的计算类型: {mission}，可选: ['relax', 'scf', 'band', 'dos']", "task_directory": task_directory, "mission": mission}}

    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}

    try:
        result = method_map[mission]()
        if isinstance(result, dict) and result.get("error"):
            return {"args": args, "returns": {"success": False, "result": result}}
        return {"args": args, "returns": {"success": True, "result": result}}
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


# ----- 机器学习工具 -----
@mcp.tool()
async def predict_band_gap(formula: str | list[str]) -> dict:
    """
    使用预训练XGBoost模型快速预测指定材料的带隙值(基于SNUMAT的HSE06泛函带隙，不适用于金属体系)，只需化学式
    
    Args:
        formula: 化学式，可以是单个字符串如 "LiFeO2" 或列表如 ["LiFeO2", "NaCl"]
    
    Returns:
        dict: 包含 formula, predicted_band_gap, error(可选)
    """
    args = {"formula": formula}
    from myml import bandgap_predict as mm
    try:
        result = mm.predict_bandgap(formula)
        return {"args": args, "returns": {"predicted_band_gap": result}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": f"预测材料 {formula} 的带隙值失败"}}


@mcp.tool()
async def predict_band_gap_gga_hse(formula: str | list[str], gap_gga: float | list[float]) -> dict:
    """
    使用 GGA→HSE 带隙修正模型预测 HSE 带隙，需要化学式 + GGA 带隙值（通常来自 Materials Project 或 OQMD 的 GGA 计算结果）

    Args:
        formula: 化学式，可以是单个字符串如 "LiFeO2" 或列表如 ["LiFeO2", "NaCl"]
        gap_gga: GGA 带隙值(eV)，可以是单个值或列表

    Returns:
        dict: 包含 predicted_hse_band_gap
    """
    args = {"formula": formula, "gap_gga": gap_gga}
    from myml import bandgap_predict as mm
    try:
        result = mm.predict_bandgap_gga_hse(formula, gap_gga)
        return {"args": args, "returns": {"predicted_hse_band_gap": result}}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": f"GGA→HSE 预测失败"}}


@mcp.tool()
async def predict_with_alignn(
    cif_path: str, 
    properties: List[str] = None,
    keep_temp_files: bool = False
) -> dict:
    """
    上传本地CIF文件到计算服务器进行ALIGNN机器学习预测(预测时间较长，建议一次预测1~3个)
    
    Args:
        cif_path: 本地CIF文件路径（将自动上传到计算服务器）
        properties: 要预测的性质列表，如 ['form_en',
    'gap_vdw',
    'elec_mass',
    'ehull',
    'gap_mbj',
    'gap_pbe',
    'hole_mass',
    'bulk_mod',
    'tot_en',
    'n_seebeck',
    'p_seebeck',
    'shear_mod',
    'encut',
    'magmom',
    'piezo_max',
    'dielectric_max',
    'mp_e_form']
                   None表示使用默认性质
                   ["all"]表示预测所有可用性质
        keep_temp_files: 是否保留远程临时文件（用于调试）
    
    Returns:
        标准MCP格式: {"args": dict, "returns": dict}
    """
    args = {"cif_path": cif_path, "keep_temp_files": keep_temp_files}
    if properties is not None:
        args["properties"] = properties

    conn_err = _check_connection()
    if conn_err:
        return {"args": args, "returns": conn_err}

    try:
        with connection as vasp_task:
            result = vasp_task.predict_from_local_cif(
                local_cif_path=cif_path,
                properties=properties,
                keep_temp=keep_temp_files
            )
            
            if result.get("status") == "ok":
                return {
                    "args": args,
                    "returns": {
                        "success": True,
                        "predictions": result.get("predictions", {}),
                        # "raw_stdout": result.get("raw_stdout", ""),
                        "raw_stderr": result.get("raw_stderr", ""),
                        "command": result.get("command", ""),
                        "upload_info": result.get("upload_info", {})
                    }
                }
            else:
                return {
                    "args": args,
                    "returns": {
                        "success": False,
                        "error": result.get("error", "预测失败"),
                        "upload_info": result.get("upload_info", {}),
                        "raw_result": result
                    }
                }
    except Exception as e:
        return {
            "args": args,
            "returns": {
                "success": False,
                "error": str(e),
                "message": "调用本地CIF预测工具失败"
            }
        }


# ============ Wyckoff 位置查询工具 ============

NUMBER_TO_SYMBOL = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2",
    6: "Pm", 7: "Pc", 8: "C2/m", 9: "C2/c", 10: "P2/m",
    11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c",
    16: "P222", 17: "P2212", 18: "P2122", 19: "P22121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I2121",
    25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21",
    30: "Pnc2", 31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2",
    35: "Cmm2", 36: "Cmc21", 37: "Ccc2", 38: "Amm2", 39: "Aem2",
    40: "Ama2", 41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2",
    45: "Iba2", 46: "Ima2",
    47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban", 51: "Pmma",
    52: "Pnna", 53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn",
    57: "Pbcm", 58: "Pnnm", 59: "Pmmn", 60: "Pbcn", 61: "Pbca",
    62: "Pnma", 63: "Cmcm", 64: "Cmce", 65: "Cmmm", 66: "Cccm",
    67: "Cmma", 68: "Ccca", 69: "Fmmm", 70: "Fddd", 71: "Immm",
    72: "Ibam", 73: "Ibca", 74: "Imma",
    75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4",
    80: "I41", 81: "P-4", 82: "I-4", 83: "P4/m", 84: "P42/m",
    85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a",
    89: "P222", 90: "P2212", 91: "P4122", 92: "P41212", 93: "P4222",
    94: "P4212", 95: "P4322", 96: "P4232", 97: "I222", 98: "I2212",
    99: "P4mm", 100: "P4bm", 101: "P42cm", 102: "P42nm", 103: "P4cc",
    104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "I4mm", 108: "I4cm",
    109: "I41md", 110: "I41cd", 111: "P-42m", 112: "P-42c", 113: "P-421m",
    114: "P-421c", 115: "P-4m2", 116: "P-4c2", 117: "P-4b2", 118: "P-4n2",
    119: "I-4m2", 120: "I-4c2", 121: "I-42m", 122: "I-42d",
    123: "P4/mmm", 124: "P4/mcc", 125: "P4/nbm", 126: "P4/nnc",
    127: "P4/mbm", 128: "P4/mnc", 129: "P4/nmm", 130: "P4/ncc",
    131: "P42/mmc", 132: "P42/mcm", 133: "P42/nbc", 134: "P42/nnm",
    135: "P42/mbc", 136: "P42/mnm", 137: "P42/nmc", 138: "P42/ncm",
    139: "I4/mmm", 140: "I4/mcm", 141: "I41/amd", 142: "I41/acd",
    143: "P3", 144: "P31", 145: "P32", 146: "R3", 147: "P-3",
    148: "R-3", 149: "P312", 150: "P321", 151: "P3112", 152: "P3121",
    153: "P3212", 154: "P3221", 155: "R32", 156: "P3m1", 157: "P31m",
    158: "P3c1", 159: "P31c", 160: "R3m", 161: "R3c", 162: "P-31m",
    163: "P-31c", 164: "P-3m1", 165: "P-3c1",
    166: "R-3m", 167: "R-3c",
    168: "P6", 169: "P61", 170: "P65", 171: "P62", 172: "P64",
    173: "P63", 174: "P-6", 175: "P6/m", 176: "P63/m", 177: "P622",
    178: "P6122", 179: "P6522", 180: "P6222", 181: "P6422", 182: "P6322",
    183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc", 187: "P-6m2",
    188: "P-6c2", 189: "P-62m", 190: "P-62c", 191: "P6/mmm", 192: "P6/mcc",
    193: "P63/mcm", 194: "P63/mmc",
    195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213",
    200: "Pm-3", 201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3",
    205: "Pa-3", 206: "Ia-3", 207: "P432", 208: "P4232", 209: "F432",
    210: "F4132", 211: "I432", 212: "P4332", 213: "P4132", 214: "I4132",
    215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c",
    220: "I-43d", 221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m",
    225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c", 229: "Im-3m",
    230: "Ia-3d",
}

_SYMBOL_TO_NUMBER = {
    "P1": 1, "P-1": 2, "P2": 3, "P21": 4, "C2": 5,
    "PM": 6, "PC": 7, "CM": 8, "CC": 9,
    "P2/M": 10, "P21/M": 11, "C2/M": 12,
    "P2/C": 13, "P21/C": 14, "C2/C": 15,
    "P222": 16, "P2212": 17, "P2122": 18, "P22121": 19, "C2221": 20,
    "C222": 21, "F222": 22, "I222": 23, "I2121": 24,
    "PMM2": 25, "PMC21": 26, "PCC2": 27, "PMA2": 28, "PCA21": 29,
    "PNC2": 30, "PMN21": 31, "PBA2": 32, "PNA21": 33, "PNN2": 34,
    "CMM2": 35, "CMC21": 36, "CCC2": 37, "AMM2": 38, "AEM2": 39,
    "AMA2": 40, "AEA2": 41, "FMM2": 42, "FDD2": 43, "IMM2": 44,
    "IBA2": 45, "IMA2": 46,
    "PMMM": 47, "PNNN": 48, "PCCM": 49, "PBAN": 50, "PMMA": 51,
    "PNNA": 52, "PMNA": 53, "PCCA": 54, "PBAM": 55, "PCCN": 56,
    "PBCM": 57, "PNNM": 58, "PMMN": 59, "PBCN": 60, "PBCA": 61,
    "PNMA": 62, "CMCM": 63, "CMCE": 64, "CMMM": 65, "CCCM": 66,
    "CMMA": 67, "CCCA": 68, "FMMM": 69, "FDDD": 70, "IMMM": 71,
    "IBAM": 72, "IBCA": 73, "IMMA": 74,
    "P4": 75, "P41": 76, "P42": 77, "P43": 78, "I4": 79, "I41": 80,
    "P-4": 81, "I-4": 82, "P4/M": 83, "P42/M": 84,
    "P4/N": 85, "P42/N": 86, "I4/M": 87, "I41/A": 88,
    "P222": 89, "P2212": 90, "P4122": 91, "P41212": 92,
    "P4222": 93, "P4212": 94, "P4322": 95, "P4232": 96,
    "I222": 97, "I2212": 98,
    "P4MM": 99, "P4BM": 100, "P42CM": 101, "P42NM": 102,
    "P4CC": 103, "P4NC": 104, "P42MC": 105, "P42BC": 106,
    "I4MM": 107, "I4CM": 108, "I41MD": 109, "I41CD": 110,
    "P-42M": 111, "P-42C": 112, "P-421M": 113, "P-421C": 114,
    "P-4M2": 115, "P-4C2": 116, "P-4B2": 117, "P-4N2": 118,
    "I-4M2": 119, "I-4C2": 120, "I-42M": 121, "I-42D": 122,
    "P4/MMM": 123, "P4/MCC": 124, "P4/NBM": 125, "P4/NNC": 126,
    "P4/MBM": 127, "P4/MNC": 128, "P4/NMM": 129, "P4/NCC": 130,
    "P42/MMC": 131, "P42/MCM": 132, "P42/NBC": 133, "P42/NNM": 134,
    "P42/MBC": 135, "P42/MNM": 136, "P42/NMC": 137, "P42/NCM": 138,
    "I4/MMM": 139, "I4/MCM": 140, "I41/AMD": 141, "I41/ACD": 142,
    "P3": 143, "P31": 144, "P32": 145, "R3": 146,
    "P-3": 147, "R-3": 148, "P312": 149, "P321": 150,
    "P3112": 151, "P3121": 152, "P3212": 153, "P3221": 154, "R32": 155,
    "P3M1": 156, "P31M": 157, "P3C1": 158, "P31C": 159,
    "R3M": 160, "R3C": 161, "P-31M": 162, "P-31C": 163,
    "P-3M1": 164, "P-3C1": 165, "R-3M": 166, "R-3C": 167,
    "P6": 168, "P61": 169, "P65": 170, "P62": 171, "P64": 172, "P63": 173,
    "P-6": 174, "P6/M": 175, "P63/M": 176,
    "P622": 177, "P6122": 178, "P6522": 179, "P6222": 180,
    "P6422": 181, "P6322": 182,
    "P6MM": 183, "P6CC": 184, "P63CM": 185, "P63MC": 186,
    "P-6M2": 187, "P-6C2": 188, "P-62M": 189, "P-62C": 190,
    "P6/MMM": 191, "P6/MCC": 192, "P63/MCM": 193, "P63/MMC": 194,
    "P23": 195, "F23": 196, "I23": 197, "P213": 198, "I213": 199,
    "PM-3": 200, "PN-3": 201, "FM-3": 202, "FD-3": 203, "IM-3": 204,
    "PA-3": 205, "IA-3": 206,
    "P432": 207, "P4232": 208, "F432": 209, "F4132": 210,
    "I432": 211, "P4332": 212, "P4132": 213, "I4132": 214,
    "P-43M": 215, "F-43M": 216, "I-43M": 217, "P-43N": 218,
    "F-43C": 219, "I-43D": 220,
    "PM-3M": 221, "PN-3N": 222, "PM-3N": 223, "PN-3M": 224,
    "FM-3M": 225, "FM-3C": 226, "FD-3M": 227, "FD-3C": 228,
    "IM-3M": 229, "IA-3D": 230,
}


def _normalize_symbol(s):
    return s.upper().replace("-", "").replace("_", "").replace(" ", "").replace("'", "")


def _crystal_system(num):
    if 1 <= num <= 2: return "三斜 (Triclinic)"
    elif 3 <= num <= 15: return "单斜 (Monoclinic)"
    elif 16 <= num <= 74: return "正交 (Orthorhombic)"
    elif 75 <= num <= 142: return "四方 (Tetragonal)"
    elif 143 <= num <= 167: return "三方 (Trigonal)"
    elif 168 <= num <= 194: return "六方 (Hexagonal)"
    elif 195 <= num <= 230: return "立方 (Cubic)"
    return "未知"


def _get_wyckoff_number(spacegroup: str) -> int | None:
    """将输入解析为空间群编号：纯数字直接转 int，否则按符号查表"""
    try:
        num = int(spacegroup)
        if 1 <= num <= 230:
            return num
        return None
    except ValueError:
        normalized = _normalize_symbol(spacegroup)
        for sym, num in _SYMBOL_TO_NUMBER.items():
            if _normalize_symbol(sym) == normalized:
                return num
        return None


def _lookup_wyckoff_data(db, number: int):
    """查找 Wyckoff 数据库：先精确匹配，再回退到带后缀的变体（如 227-1）"""
    num_str = str(number)
    if num_str in db.data:
        return db.data[num_str]
    # 尝试第一个原点选择
    if f"{num_str}-1" in db.data:
        return db.data[f"{num_str}-1"]
    # 尝试任何以该编号开头的键
    for key in db.data:
        if key.startswith(num_str + "-"):
            return db.data[key]
    return None


@mcp.tool()
async def get_wyckoff_positions(spacegroup: str) -> dict:
    """
    根据空间群编号 (1-230) 或国际符号 (如 "Fm-3m", "P42/mnm") 获取 Wyckoff 位置信息

    Args:
        spacegroup: 空间群编号（如 "225"）或国际符号（如 "Fm-3m"，大小写不敏感）

    Returns:
        dict: 包含 spacegroup_number, symbol, crystal_system, wyckoff_positions
    """
    import wyckoff
    from fractions import Fraction

    number = _get_wyckoff_number(spacegroup)
    if number is None:
        return {"args": {"spacegroup": spacegroup}, "returns": {"error": f"无法识别的空间群: '{spacegroup}'，请输入 1-230 的编号或标准国际符号"}}

    symbol = NUMBER_TO_SYMBOL.get(number, str(number))

    def _fmt(val):
        if isinstance(val, Fraction):
            return f"{float(val):.6g}"
        return str(val)

    db = wyckoff.WyckoffDatabase()
    sg_data = _lookup_wyckoff_data(db, number)
    if not sg_data:
        return {"args": {"spacegroup": spacegroup}, "returns": {"error": f"数据库中未找到空间群 {number}"}}

    positions = []
    for wp in sg_data.wyckoff_positions:
        if wp.coordinates:
            coord = wp.coordinates[0]
            coord_str = f"({_fmt(coord[0])}, {_fmt(coord[1])}, {_fmt(coord[2])})"
        else:
            coord_str = None
        positions.append({
            "label": wp.label,
            "multiplicity": wp.multiplicity,
            "site_symmetry": wp.site_symmetry,
            "coordinates": coord_str,
        })

    return {
        "args": {"spacegroup": spacegroup},
        "returns": {
            "spacegroup_number": number,
            "symbol": symbol,
            "crystal_system": _crystal_system(number),
            "wyckoff_positions": positions,
        }
    }


# ============ 主程序入口 ============
if __name__ == "__main__":
    try:
        # 启动文件服务器
        matfileserver = flask_server.MatFileServer()
        # 连接远程服务器（失败不影响其他工具）
        connection = None
        try:
            connection = tryssh.VaspTaskInitializer(HOST, USERNAME, PASSWORD, PORT)
            for i in range(5):
                try:
                    with connection as vasp_task:
                        if vasp_task.link():
                            print("已成功连接到远程服务器")
                            break
                except Exception as e:
                    print(f"连接远程服务器失败，正在重试... ({i+1}/5), 错误: {e}")
                    if i == 4:
                        print("⚠️ 无法连接到计算服务器，VASP/计算相关工具将不可用，其他工具正常服务")
                        connection = None
        except Exception as e:
            print(f"⚠️ 计算服务器初始化失败: {e}，VASP/计算相关工具将不可用，其他工具正常服务")
            connection = None

        # 启动 MCP 服务器
        mcp.run(
            transport="sse",
            host="0.0.0.0",
            port=8000
        )
    except Exception as e:
        print(f"服务器运行出错: {e}")
        exit()
