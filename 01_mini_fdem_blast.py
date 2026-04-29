import numpy as np
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. 物理与几何参数设置
    # ==========================================
    nx, ny = 31, 31          # 节点网格数 (31x31)，中心点为 (15, 15)
    L_total = 1.0            # 材料总边长 (m)
    dx = L_total / (nx - 1)  # 节点间距 (即弹簧初始长度)
    
    mass = 0.1               # 每个节点的质量 (kg)
    k_spring = 5.0e7         # 弹簧刚度 (N/m)，代表材料的弹性模量
    strain_fail = 0.02       # 临界断裂应变 (2%)，超过此拉应变则弹簧断裂
    
    dt = 1.0e-5              # 时间步长 (s)，显式计算需要足够小以保证稳定
    total_steps = 400        # 总计算步数
    
    # ==========================================
    # 2. 初始化节点 (Nodes)
    # ==========================================
    num_nodes = nx * ny
    pos = np.zeros((num_nodes, 2))  # 节点坐标 [x, y]
    vel = np.zeros((num_nodes, 2))  # 节点速度 [vx, vy]
    
    # 生成均匀网格坐标
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            pos[idx, 0] = i * dx
            pos[idx, 1] = j * dx
            
    # ==========================================
    # 3. 初始化弹簧 (Springs) - 构建材料连续性
    # ==========================================
    springs = []
    # 建立相邻节点的连接（水平、垂直和对角线）
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            # 连接右侧节点
            if i < nx - 1: springs.append([idx, (i + 1) * ny + j])
            # 连接上方节点
            if j < ny - 1: springs.append([idx, i * ny + j + 1])
            # 连接右上对角节点
            if i < nx - 1 and j < ny - 1: springs.append([idx, (i + 1) * ny + j + 1])
            # 连接右下对角节点
            if i < nx - 1 and j > 0: springs.append([idx, (i + 1) * ny + j - 1])
            
    springs = np.array(springs)
    num_springs = len(springs)
    
    # 计算弹簧的初始长度 L0
    p1 = pos[springs[:, 0]]
    p2 = pos[springs[:, 1]]
    L0 = np.linalg.norm(p2 - p1, axis=1)
    
    # 弹簧状态标签：True代表完好，False代表已断裂 (裂纹)
    active_springs = np.ones(num_springs, dtype=bool) 
    
    # 保存初始状态用于画图
    pos_init = pos.copy()
    active_init = active_springs.copy()

    # ==========================================
    # 4. 施加中心爆炸冲击载荷
    # ==========================================
    # 找到中心点及其周围的几个节点，赋予初始向外的脉冲速度（模拟爆轰气体膨胀）
    center_idx = (nx // 2) * ny + (ny // 2)
    center_pos = pos[center_idx]
    
    blast_radius = dx * 1.5
    blast_velocity_magnitude = 150.0  # 爆炸产生的初始强脉冲速度 (m/s)
    
    for i in range(num_nodes):
        dist_to_center = np.linalg.norm(pos[i] - center_pos)
        if dist_to_center < blast_radius:
            # 速度方向沿着径向向外
            direction = pos[i] - center_pos
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
                vel[i] = direction * blast_velocity_magnitude

    # ==========================================
    # 5. 显式时间积分主循环 (核心动力学计算)
    # ==========================================
    print("开始爆炸冲击计算...")
    for step in range(total_steps):
        # 每次循环前，节点受力清零
        force = np.zeros((num_nodes, 2))
        
        # 仅计算尚未断裂的弹簧
        valid_idx = active_springs
        if not np.any(valid_idx):
            break # 全部断裂则停止
            
        node1 = springs[valid_idx, 0]
        node2 = springs[valid_idx, 1]
        
        # 当前节点位置
        p1 = pos[node1]
        p2 = pos[node2]
        
        # 弹簧当前长度与方向向量
        vec = p2 - p1
        L_current = np.linalg.norm(vec, axis=1)
        
        # 计算应变
        L0_valid = L0[valid_idx]
        strain = (L_current - L0_valid) / L0_valid
        
        # >>> 断裂准则 <<<
        # 找出当前步拉断的弹簧，并将其状态置为 False
        broken_this_step = strain >= strain_fail
        if np.any(broken_this_step):
            # 将 valid_idx 中对应断裂的位置设为 False
            active_springs[np.where(valid_idx)[0][broken_this_step]] = False
            
        # 过滤掉刚刚断裂的弹簧，只计算仍然完好的弹簧力
        still_valid = ~broken_this_step
        n1 = node1[still_valid]
        n2 = node2[still_valid]
        L_cur = L_current[still_valid]
        L0_val = L0_valid[still_valid]
        v = vec[still_valid]
        
        # 胡克定律计算弹簧力大小: F = k * (L - L0)
        f_mag = k_spring * (L_cur - L0_val)
        
        # 计算弹簧力向量，并累加到对应的节点上
        f_vec = (v / (L_cur[:, np.newaxis] + 1e-9)) * f_mag[:, np.newaxis]
        
        # 使用 np.add.at 解决重复索引累加问题
        np.add.at(force, n1, f_vec)    # 节点1受力
        np.add.at(force, n2, -f_vec)   # 节点2受反作用力
        
        # 更新速度和位移 (简单的显式 Euler 方法)
        acc = force / mass
        vel += acc * dt
        pos += vel * dt
        
        if step % 50 == 0:
            print(f"计算进度: {step}/{total_steps} 步 | 裂纹(断裂弹簧)数量: {num_springs - np.sum(active_springs)}")
            
    print("计算完成！正在生成对比图像...")

    # ==========================================
    # 6. 可视化绘制 (初始状态 vs 破坏状态)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制函数
    def plot_mesh(ax, p, spr, active, title, color_node, color_line):
        ax.set_title(title, fontsize=14)
        ax.scatter(p[:, 0], p[:, 1], c=color_node, s=10, zorder=3)
        # 画线段
        valid_spr = spr[active]
        for s in valid_spr:
            pt1, pt2 = p[s[0]], p[s[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color_line, lw=0.5, zorder=1)
        ax.set_aspect('equal')
        ax.axis('off')

    # 图1：初始状态
    plot_mesh(axes[0], pos_init, springs, active_init, "Initial State (Continuum)", 'black', 'gray')
    
    # 图2：破坏后状态 (只画出仍然完好的弹簧，消失的线段即为裂纹)
    plot_mesh(axes[1], pos, springs, active_springs, "Post-Blast Fracture State", 'red', 'blue')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()