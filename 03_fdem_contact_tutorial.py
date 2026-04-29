import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def main():
    # ==========================================
    # 1. 物理与几何参数设置
    # ==========================================
    mass = 0.1               # 节点质量 (kg)
    k_spring = 5.0e7         # 内部弹簧刚度 (保持块体形状)
    
    # --- 新增：接触力学参数 ---
    contact_radius = 0.05    # 节点的有效碰撞半径 (m)
    k_penalty = 1.0e8        # 惩罚接触刚度 (通常比内部刚度更大，防止穿透)
    
    dt = 1.0e-5              # 时间步长 (s)
    total_steps = 1500       # 总计算步数

    # ==========================================
    # 2. 初始化两个独立的块体 (Block A 和 Block B)
    # ==========================================
    nx, ny = 3, 3
    dx = 0.08  
    
    # 生成 Block A
    pos_A = np.zeros((nx * ny, 2))
    for i in range(nx):
        for j in range(ny):
            pos_A[i * ny + j] = [i * dx, j * dx]
    vel_A = np.ones((nx * ny, 2)) * np.array([20.0, 0.0]) 
            
    # 生成 Block B 
    pos_B = np.zeros((nx * ny, 2))
    offset_x = 0.6 
    for i in range(nx):
        for j in range(ny):
            pos_B[i * ny + j] = [i * dx + offset_x, j * dx + 0.02] 
    vel_B = np.ones((nx * ny, 2)) * np.array([-20.0, 0.0]) 
    
    # 合并节点系统
    pos = np.vstack((pos_A, pos_B))
    vel = np.vstack((vel_A, vel_B))
    num_nodes = len(pos)
    
    # ==========================================
    # 3. 建立内部弹簧 
    # ==========================================
    springs = []
    def build_springs(start_idx):
        for i in range(nx):
            for j in range(ny):
                idx = start_idx + i * ny + j
                if i < nx - 1: springs.append([idx, start_idx + (i + 1) * ny + j])
                if j < ny - 1: springs.append([idx, start_idx + i * ny + j + 1])
                if i < nx - 1 and j < ny - 1: springs.append([idx, start_idx + (i + 1) * ny + j + 1])
                if i < nx - 1 and j > 0: springs.append([idx, start_idx + (i + 1) * ny + j - 1])
                
    build_springs(0)             
    build_springs(nx * ny)       
    
    springs = np.array(springs)
    p1_init, p2_init = pos[springs[:, 0]], pos[springs[:, 1]]
    L0 = np.linalg.norm(p2_init - p1_init, axis=1)

    history_pos = []
    record_steps = [0, 600, 1400] 

    # ==========================================
    # 4. 显式动力学 + 接触判定主循环
    # ==========================================
    print("开始双块体碰撞接触计算...")
    for step in range(total_steps):
        force = np.zeros((num_nodes, 2))
        
        # --- a. 计算内部弹簧力 ---
        n1, n2 = springs[:, 0], springs[:, 1]
        p1, p2 = pos[n1], pos[n2]
        vec = p2 - p1
        L_cur = np.linalg.norm(vec, axis=1)
        f_mag = k_spring * (L_cur - L0)
        f_vec = (vec / (L_cur[:, np.newaxis] + 1e-9)) * f_mag[:, np.newaxis]
        np.add.at(force, n1, f_vec)
        np.add.at(force, n2, -f_vec)
        
        # --- b. 计算节点间的接触力 ---
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist_vec = pos[i] - pos[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = contact_radius * 2
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist  
                    repulsive_force_mag = k_penalty * overlap
                    repulsive_dir = dist_vec / dist
                    force[i] += repulsive_force_mag * repulsive_dir
                    force[j] -= repulsive_force_mag * repulsive_dir
        
        # --- c. 显式时间积分 ---
        acc = force / mass
        vel += acc * dt
        pos += vel * dt
        
        if step in record_steps:
            history_pos.append(pos.copy())
            print(f"记录关键帧数据: 第 {step} 步")

    print("计算完成，正在生成接触碰撞全过程图...")

    # ==========================================
    # 5. 可视化绘制与自动保存
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["1. Before Impact", "2. During Contact (Penalty Activated)", "3. Rebound (After Collision)"]
    
    def plot_frame(ax, p, title):
        ax.set_title(title, fontsize=14, pad=15)
        for i in range(num_nodes):
            circle = plt.Circle((p[i, 0], p[i, 1]), contact_radius, color='lightgray', alpha=0.4, zorder=1)
            ax.add_patch(circle)
            
        ax.scatter(p[:9, 0], p[:9, 1], c='blue', s=30, label='Block A', zorder=3)
        ax.scatter(p[9:, 0], p[9:, 1], c='red', s=30, label='Block B', zorder=3)
        
        lines = [p[s] for s in springs]
        from matplotlib.collections import LineCollection
        lc = LineCollection(lines, colors='black', linewidths=1.0, alpha=0.6, zorder=2)
        ax.add_collection(lc)
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        if title == titles[0]: ax.legend(loc='upper right')

    for idx, ax in enumerate(axes):
        plot_frame(ax, history_pos[idx], titles[idx])
        
    plt.tight_layout()

    save_dir = r"C:\Users\Administrator\Desktop\Gemini_FDEM"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%H%M%S")
    save_path = os.path.join(save_dir, f"03_Contact_Penalty_Method_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图像已成功保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()