import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
import os
from datetime import datetime
import time

def main():
    # ==========================================
    # 1. 物理与几何参数设置
    # ==========================================
    L_total = 1.0
    num_random_nodes = 800   # 节点数适中，保证向量化接触搜索能在几十秒内算完
    num_boundary_nodes = 30
    
    mass = 0.05
    k_spring = 8.0e7         # 混凝土弹簧刚度
    strain_fail = 0.015      # 临界断裂应变
    
    # 接触力学参数
    # 估算平均节点间距 dx，接触半径应略小于 dx/2
    contact_radius = 0.014   
    k_penalty = 1.5e8        # 接触刚度略大于弹簧刚度，防止高速穿透
    
    dt = 4.0e-6              # 引入接触后，体系更容易高频振荡，时间步需进一步缩小
    total_steps = 1000       # 延长计算步数，让碎块有足够时间飞溅和相互撞击

    # ==========================================
    # 2. 生成非结构化网格
    # ==========================================
    print("正在生成非结构化网格...")
    internal_pts = np.random.rand(num_random_nodes, 2) * L_total
    bx = np.linspace(0, L_total, num_boundary_nodes)
    zero_arr, full_arr = np.zeros_like(bx), np.ones_like(bx) * L_total
    pos = np.vstack((internal_pts, 
                     np.column_stack((bx, zero_arr)), np.column_stack((bx, full_arr)),
                     np.column_stack((zero_arr, bx)), np.column_stack((full_arr, bx))))
    pos = np.unique(pos, axis=0)
    num_nodes = len(pos)
    vel = np.zeros((num_nodes, 2))
    
    # Delaunay 剖分生成弹簧
    tri = Delaunay(pos)
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted([simplex[0], simplex[1]])))
        edges.add(tuple(sorted([simplex[1], simplex[2]])))
        edges.add(tuple(sorted([simplex[2], simplex[0]])))
    springs = np.array(list(edges))
    num_springs = len(springs)
    
    p1, p2 = pos[springs[:, 0]], pos[springs[:, 1]]
    L0 = np.linalg.norm(p2 - p1, axis=1)
    active_springs = np.ones(num_springs, dtype=bool)
    
    pos_init = pos.copy()
    active_init = active_springs.copy()

    # ==========================================
    # 3. 施加多孔爆破冲击 (三孔配置)
    # ==========================================
    boreholes = np.array([[0.5, 0.65], [0.35, 0.4], [0.65, 0.4]])
    blast_radius = 0.08
    blast_vel = 200.0  # 稍微增大脉冲速度，让碎块飞得更明显
    
    for bh in boreholes:
        for i in range(num_nodes):
            dist = np.linalg.norm(pos[i] - bh)
            if dist < blast_radius:
                dir_vec = pos[i] - bh
                if np.linalg.norm(dir_vec) > 1e-6:
                    vel[i] += (dir_vec / np.linalg.norm(dir_vec)) * blast_vel

    history_pos, history_active = [], []
    record_steps = [0, 300, 999] # 记录：初始、开始破裂、最终碎块飞溅

    # ==========================================
    # 4. 显式动力学 + 动态断裂与接触耦合
    # ==========================================
    print(f"开始耦合动力学计算... 节点数:{num_nodes}, 弹簧数:{num_springs}")
    start_time = time.time()
    
    for step in range(total_steps):
        force = np.zeros((num_nodes, 2))
        
        # ----------------------------------------
        # A. 计算内部弹簧力与断裂准则
        # ----------------------------------------
        valid_idx = active_springs
        if np.any(valid_idx):
            n1, n2 = springs[valid_idx, 0], springs[valid_idx, 1]
            p1, p2 = pos[n1], pos[n2]
            vec = p2 - p1
            L_cur = np.linalg.norm(vec, axis=1)
            L0_val = L0[valid_idx]
            strain = (L_cur - L0_val) / L0_val
            
            # 断裂判定
            broken_this_step = strain >= strain_fail
            if np.any(broken_this_step):
                # 将断裂的弹簧标记为 False
                active_springs[np.where(valid_idx)[0][broken_this_step]] = False
                
            # 过滤掉刚断裂的，只算依然完好的内力
            still_valid = ~broken_this_step
            n1, n2 = n1[still_valid], n2[still_valid]
            L_c, L0_v = L_cur[still_valid], L0_val[still_valid]
            v = vec[still_valid]
            
            f_mag = k_spring * (L_c - L0_v)
            f_vec = (v / (L_c[:, np.newaxis] + 1e-9)) * f_mag[:, np.newaxis]
            
            np.add.at(force, n1, f_vec)
            np.add.at(force, n2, -f_vec)

        # ----------------------------------------
        # B. 高效向量化全局接触搜索 (核心进阶)
        # ----------------------------------------
        # 1. 动态生成邻接矩阵 (记录当前时间步哪些节点还连着)
        adj = np.zeros((num_nodes, num_nodes), dtype=bool)
        curr_valid = springs[active_springs]
        # 对称赋值：如果 i,j 相连，则 adj[i,j] 和 adj[j,i] 均为 True
        adj[curr_valid[:,0], curr_valid[:,1]] = True
        adj[curr_valid[:,1], curr_valid[:,0]] = True

        # 2. 计算所有节点两两之间的距离矩阵 (N x N)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] 
        dist = np.linalg.norm(diff, axis=-1)
        
        # 使用 triu 提取上三角矩阵，避免重复计算 (i碰j 和 j碰i)
        dist = np.triu(dist, k=1) 
        
        # 3. 接触过滤条件：距离 < 2*R 且 节点之间没有连接 (adj == False)
        min_dist = contact_radius * 2
        contact_mask = (dist < min_dist) & (dist > 1e-6) & (~adj)
        
        # 4. 计算并施加接触力
        if np.any(contact_mask):
            i_idx, j_idx = np.where(contact_mask)
            overlap = min_dist - dist[i_idx, j_idx]
            repulsive_mag = k_penalty * overlap
            
            # 力的方向从 j 指向 i
            direction = diff[i_idx, j_idx, :] / dist[i_idx, j_idx, np.newaxis]
            f_cont = direction * repulsive_mag[:, np.newaxis]
            
            np.add.at(force, i_idx, f_cont)
            np.add.at(force, j_idx, -f_cont)

        # ----------------------------------------
        # C. 显式积分更新位移
        # ----------------------------------------
        acc = force / mass
        vel += acc * dt
        pos += vel * dt
        
        if step in record_steps:
            history_pos.append(pos.copy())
            history_active.append(active_springs.copy())
            
        if step % 100 == 0:
            print(f"进度: {step:04d}/{total_steps} | 断裂数: {num_springs - np.sum(active_springs)} | 接触对数量: {np.sum(contact_mask)}")

    print(f"计算完成！耗时: {time.time() - start_time:.2f} 秒")

    # ==========================================
    # 5. 可视化绘制与自动保存
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["1. Initial State", "2. Fracture Initiation", "3. Fragmentation & Contact Interaction"]
    
    def plot_frame(ax, p, spr, active, title):
        ax.set_title(title, fontsize=14, pad=15)
        # 只画节点，颜色根据状态稍微变化
        ax.scatter(p[:, 0], p[:, 1], c='black', s=8, alpha=0.8, zorder=3)
        
        # 画线
        valid_spr = spr[active]
        lines = [p[s] for s in valid_spr]
        lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.6, zorder=1)
        ax.add_collection(lc)
        
        ax.set_aspect('equal')
        # 视角稍微拉远，以看清飞出去的碎块
        ax.set_xlim(-0.2, L_total + 0.2)
        ax.set_ylim(-0.2, L_total + 0.2)
        ax.axis('off')

    for idx, ax in enumerate(axes):
        plot_frame(ax, history_pos[idx], springs, history_active[idx], titles[idx])
        if idx == 0: # 在初始图上标出孔
            for bh in boreholes:
                ax.add_patch(plt.Circle((bh[0], bh[1]), blast_radius*0.3, color='orange', alpha=0.8))
        
    plt.tight_layout()

    save_dir = r"C:\Users\Administrator\Desktop\Gemini_FDEM"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%H%M%S")
    save_path = os.path.join(save_dir, f"04_Coupled_Fracture_Contact_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图像已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()