import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
import os
from datetime import datetime

def main():
    # ==========================================
    # 1. 物理与几何参数设置
    # ==========================================
    L_total = 1.0            # 材料总边长 (m)
    num_random_nodes = 1200  # 随机节点数量
    num_boundary_nodes = 40  # 每条边界上的节点数 (保证边界平滑)
    
    mass = 0.05              # 节点质量 (kg)
    k_spring = 8.0e7         # 弹簧刚度 (N/m) - 模拟混凝土较高的弹性模量
    strain_fail = 0.015      # 临界断裂应变 (1.5%) - 模拟混凝土的脆性
    
    dt = 5.0e-6              # 时间步长 (s) - 非结构化网格局部较密，需更小的时间步以保证稳定
    total_steps = 600        # 总计算步数

    # ==========================================
    # 2. 生成非结构化随机节点 (Nodes)
    # ==========================================
    print("正在生成非结构化网格...")
    # 2.1 内部随机节点
    internal_pts = np.random.rand(num_random_nodes, 2) * L_total
    
    # 2.2 边界规则节点 (为了让材料外轮廓是平直的正方形)
    bx = np.linspace(0, L_total, num_boundary_nodes)
    zero_arr = np.zeros_like(bx)
    full_arr = np.ones_like(bx) * L_total
    
    bottom = np.column_stack((bx, zero_arr))
    top    = np.column_stack((bx, full_arr))
    left   = np.column_stack((zero_arr, bx))
    right  = np.column_stack((full_arr, bx))
    
    # 合并所有节点
    pos = np.vstack((internal_pts, bottom, top, left, right))
    # 去除重复点 (角落的点会重复)
    pos = np.unique(pos, axis=0)
    num_nodes = len(pos)
    
    vel = np.zeros((num_nodes, 2))  # 初始化速度
    
    # ==========================================
    # 3. Delaunay 三角剖分生成弹簧 (Springs)
    # ==========================================
    tri = Delaunay(pos)
    
    # 从三角形网格中提取唯一的边作为弹簧
    edges = set()
    for simplex in tri.simplices:
        # 对三角形的三个顶点排序并组合，确保边不重复
        edges.add(tuple(sorted([simplex[0], simplex[1]])))
        edges.add(tuple(sorted([simplex[1], simplex[2]])))
        edges.add(tuple(sorted([simplex[2], simplex[0]])))
        
    springs = np.array(list(edges))
    num_springs = len(springs)
    
    # 计算弹簧初始长度
    p1 = pos[springs[:, 0]]
    p2 = pos[springs[:, 1]]
    L0 = np.linalg.norm(p2 - p1, axis=1)
    
    # 弹簧状态：True 为完好
    active_springs = np.ones(num_springs, dtype=bool)
    
    pos_init = pos.copy()
    active_init = active_springs.copy()

    # ==========================================
    # 4. 施加多孔爆破冲击载荷 (三孔配置)
    # ==========================================
    # 设定三个爆破孔的坐标 (呈三角形排布)
    boreholes = np.array([
        [0.5, 0.65],  # 顶部孔
        [0.35, 0.4],  # 左下孔
        [0.65, 0.4]   # 右下孔
    ])
    
    blast_radius = 0.08  # 爆破孔影响半径
    blast_vel = 180.0    # 爆破脉冲速度 (m/s)
    
    for bh in boreholes:
        for i in range(num_nodes):
            dist = np.linalg.norm(pos[i] - bh)
            if dist < blast_radius:
                dir_vec = pos[i] - bh
                if np.linalg.norm(dir_vec) > 1e-6:
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    vel[i] += dir_vec * blast_vel # 速度叠加，模拟波的交汇处更强

    # ==========================================
    # 5. 显式动力学主循环
    # ==========================================
    print(f"开始三孔爆破动力学计算，总节点数:{num_nodes}，总弹簧数:{num_springs}")
    for step in range(total_steps):
        force = np.zeros((num_nodes, 2))
        valid_idx = active_springs
        
        if not np.any(valid_idx):
            break
            
        node1 = springs[valid_idx, 0]
        node2 = springs[valid_idx, 1]
        
        p1 = pos[node1]
        p2 = pos[node2]
        
        vec = p2 - p1
        L_current = np.linalg.norm(vec, axis=1)
        
        L0_valid = L0[valid_idx]
        strain = (L_current - L0_valid) / L0_valid
        
        # 断裂准则
        broken_this_step = strain >= strain_fail
        if np.any(broken_this_step):
            active_springs[np.where(valid_idx)[0][broken_this_step]] = False
            
        still_valid = ~broken_this_step
        n1 = node1[still_valid]
        n2 = node2[still_valid]
        L_cur = L_current[still_valid]
        L0_val = L0_valid[still_valid]
        v = vec[still_valid]
        
        # 胡克定律
        f_mag = k_spring * (L_cur - L0_val)
        f_vec = (v / (L_cur[:, np.newaxis] + 1e-9)) * f_mag[:, np.newaxis]
        
        np.add.at(force, n1, f_vec)
        np.add.at(force, n2, -f_vec)
        
        # 显式积分
        acc = force / mass
        vel += acc * dt
        pos += vel * dt
        
        if step % 100 == 0:
            print(f"当前步: {step:03d}/{total_steps} | 宏观裂纹(断裂弹簧数): {num_springs - np.sum(active_springs)}")
            
    print("计算完成，正在渲染图像并保存...")

    # ==========================================
    # 6. 可视化与自动保存模块
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    def plot_mesh(ax, p, spr, active, title, color_node, color_line):
        ax.set_title(title, fontsize=16, pad=15)
        # 标记三个爆破孔的位置
        for bh in boreholes:
            ax.add_patch(plt.Circle((bh[0], bh[1]), blast_radius*0.3, color='orange', zorder=4, alpha=0.8))
            
        ax.scatter(p[:, 0], p[:, 1], c=color_node, s=5, alpha=0.8, zorder=3)
        valid_spr = spr[active]
        lines = [p[s] for s in valid_spr]
        lc = LineCollection(lines, colors=color_line, linewidths=0.5, alpha=0.7, zorder=1)
        ax.add_collection(lc)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, L_total + 0.05)
        ax.set_ylim(-0.05, L_total + 0.05)
        ax.axis('off')

    plot_mesh(axes[0], pos_init, springs, active_init, "Step 02: Unstructured Delaunay Mesh\n(Three-Borehole Setup)", 'black', 'gray')
    plot_mesh(axes[1], pos, springs, active_springs, "Fracture Network & Crack Branching\n(Wave Interaction Zone)", 'darkred', 'steelblue')
    
    plt.tight_layout()

    # --- 按照要求自动保存到指定目录 ---
    save_dir = r"C:\Users\Administrator\Desktop\Gemini_FDEM"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.now().strftime("%H%M%S")
    file_name = f"02_Unstructured_3Borehole_Blast_{timestamp}.png"
    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图像已成功保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()